//! Abstract domains for dataflow analysis.
//!
//! Provides `SemiLattice` (the fixpoint algebra) and `AbstractValue` (the
//! concrete domain used by forward value analysis / branch pruning).
//!
//! # AbstractValue lattice
//!
//! ```text
//!   Top              "could be anything"
//!    |
//!  Finite(...)       "one of these concrete values"
//!    |
//!  Bottom            "unreachable / no value"
//! ```
//!
//! `Finite` wraps a `FiniteSet` — a bounded collection of concrete values.
//! When the set grows beyond `MAX_SET_SIZE`, the value widens to `Top`.
//! This guarantees termination: every chain Bottom → Finite → Top is finite.
//!
//! # FiniteSet variants
//!
//! | Variant      | Represents                | Widening          |
//! |--------------|---------------------------|-------------------|
//! | `Intervals`  | Int values as intervals   | graduated merging |
//! | `Bools`      | {true}, {false}, or both  | at most 2         |
//! | `Variants`   | tagged union possibilities| per-tag payload   |
//! | `Literals`   | Float, Byte, String, List | set union to MAX  |
//! | `Tuple`      | per-element abstract val  | element-wise join |
//!
//! Int gets its own variant (`Intervals`) because interval arithmetic enables
//! graduated widening — merging closest pairs instead of jumping to Top.
//! All other scalar types use `Literals` (flat set, no arithmetic).

use acvus_ast::Literal;
use acvus_utils::Astr;
use smallvec::SmallVec;

use crate::analysis::reachable_context::KnownValue;

// ── SemiLattice ────────────────────────────────────────────────────

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

// ── Interval ───────────────────────────────────────────────────────

/// Closed integer interval [lo, hi]. A point value is `Interval { lo: n, hi: n }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Interval {
    pub lo: i64,
    pub hi: i64,
}

impl Interval {
    pub fn point(v: i64) -> Self {
        Interval { lo: v, hi: v }
    }

    pub fn contains(&self, v: i64) -> bool {
        v >= self.lo && v <= self.hi
    }

    pub fn overlaps_or_adjacent(&self, other: &Interval) -> bool {
        self.lo <= other.hi.saturating_add(1) && other.lo <= self.hi.saturating_add(1)
    }

    pub fn merge(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    pub fn gap_to(&self, other: &Interval) -> u64 {
        if self.overlaps_or_adjacent(other) {
            0
        } else if self.hi < other.lo {
            (other.lo - self.hi) as u64
        } else {
            (self.lo - other.hi) as u64
        }
    }
}

// ── AbstractValue / FiniteSet ──────────────────────────────────────

/// Maximum number of elements in a FiniteSet before widening to Top.
///
/// 16 balances precision vs cost: branch pruning rarely needs more than
/// a handful of values, and interval widening degrades gracefully.
const MAX_SET_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub enum AbstractValue {
    Bottom,
    Finite(FiniteSet),
    Top,
}

/// Bounded concrete value set. See module doc for variant descriptions.
#[derive(Debug, Clone, PartialEq)]
pub enum FiniteSet {
    Intervals(SmallVec<[Interval; 4]>),
    Bools(SmallVec<[bool; 2]>),
    Variants(SmallVec<[(Astr, Box<AbstractValue>); 4]>),
    Literals(SmallVec<[Literal; 4]>),
    Tuple(Vec<AbstractValue>),
}

// ── SemiLattice for AbstractValue ──────────────────────────────────

impl SemiLattice for AbstractValue {
    fn bottom() -> Self {
        AbstractValue::Bottom
    }

    fn join_mut(&mut self, other: &Self) -> bool {
        match (&*self, other) {
            (_, AbstractValue::Bottom) => false,
            (AbstractValue::Top, _) => false,
            (AbstractValue::Bottom, _) => {
                *self = other.clone();
                true
            }
            (_, AbstractValue::Top) => {
                *self = AbstractValue::Top;
                true
            }
            (AbstractValue::Finite(a), AbstractValue::Finite(b)) => {
                let merged = join_finite_sets(a, b);
                if *self != merged {
                    *self = merged;
                    true
                } else {
                    false
                }
            }
        }
    }
}

// ── FiniteSet join ─────────────────────────────────────────────────

fn join_finite_sets(a: &FiniteSet, b: &FiniteSet) -> AbstractValue {
    match (a, b) {
        (FiniteSet::Intervals(ai), FiniteSet::Intervals(bi)) => join_intervals(ai, bi),
        (FiniteSet::Bools(ab), FiniteSet::Bools(bb)) => {
            let mut merged: SmallVec<[bool; 2]> = ab.clone();
            for &v in bb {
                if !merged.contains(&v) {
                    merged.push(v);
                }
            }
            merged.sort();
            AbstractValue::Finite(FiniteSet::Bools(merged))
        }
        (FiniteSet::Variants(av), FiniteSet::Variants(bv)) => join_variants(av, bv),
        (FiniteSet::Literals(al), FiniteSet::Literals(bl)) => {
            let mut merged: SmallVec<[Literal; 4]> = al.clone();
            for v in bl {
                if !merged.iter().any(|x| literal_eq(x, v)) {
                    if merged.len() >= MAX_SET_SIZE {
                        return AbstractValue::Top;
                    }
                    merged.push(v.clone());
                }
            }
            AbstractValue::Finite(FiniteSet::Literals(merged))
        }
        (FiniteSet::Tuple(at), FiniteSet::Tuple(bt)) => {
            if at.len() != bt.len() {
                return AbstractValue::Top;
            }
            let mut elems = at.clone();
            for (e, other) in elems.iter_mut().zip(bt.iter()) {
                e.join_mut(other);
            }
            AbstractValue::Finite(FiniteSet::Tuple(elems))
        }
        // Different FiniteSet kinds → incompatible types → Top.
        _ => AbstractValue::Top,
    }
}

/// Join two interval sets with graduated widening.
///
/// Instead of jumping straight to Top when over MAX_SET_SIZE, repeatedly
/// merge the closest pair of intervals. This preserves range information
/// (e.g., two separate ranges [1,3] and [7,9] stay separate until forced).
fn join_intervals(a: &[Interval], b: &[Interval]) -> AbstractValue {
    let mut all: SmallVec<[Interval; 8]> = SmallVec::new();
    all.extend_from_slice(a);
    all.extend_from_slice(b);
    all.sort_by(|x, y| x.lo.cmp(&y.lo).then(x.hi.cmp(&y.hi)));

    // Merge overlapping/adjacent intervals.
    let mut merged: SmallVec<[Interval; 4]> = SmallVec::new();
    for iv in all {
        if let Some(last) = merged.last_mut()
            && last.overlaps_or_adjacent(&iv)
        {
            *last = last.merge(&iv);
            continue;
        }
        merged.push(iv);
    }

    // Graduated widening: merge closest pairs until within limit.
    while merged.len() > MAX_SET_SIZE && merged.len() >= 2 {
        let mut min_gap = u64::MAX;
        let mut min_idx = 0;
        for i in 0..merged.len() - 1 {
            let gap = merged[i].gap_to(&merged[i + 1]);
            if gap < min_gap {
                min_gap = gap;
                min_idx = i;
            }
        }
        let combined = merged[min_idx].merge(&merged[min_idx + 1]);
        merged[min_idx] = combined;
        merged.remove(min_idx + 1);
    }

    AbstractValue::Finite(FiniteSet::Intervals(merged))
}

fn join_variants(
    a: &[(Astr, Box<AbstractValue>)],
    b: &[(Astr, Box<AbstractValue>)],
) -> AbstractValue {
    let mut merged: SmallVec<[(Astr, Box<AbstractValue>); 4]> = a.into();
    for (tag, payload) in b {
        if let Some(entry) = merged.iter_mut().find(|(t, _)| t == tag) {
            entry.1.join_mut(payload);
        } else {
            if merged.len() >= MAX_SET_SIZE {
                return AbstractValue::Top;
            }
            merged.push((*tag, payload.clone()));
        }
    }
    AbstractValue::Finite(FiniteSet::Variants(merged))
}

/// Structural equality for `Literal`. Needed because `f64` doesn't impl `Eq`.
/// Uses `to_bits()` for floats so NaN == NaN (same bit pattern).
fn literal_eq(a: &Literal, b: &Literal) -> bool {
    match (a, b) {
        (Literal::Int(a), Literal::Int(b)) => a == b,
        (Literal::Bool(a), Literal::Bool(b)) => a == b,
        (Literal::String(a), Literal::String(b)) => a == b,
        (Literal::Byte(a), Literal::Byte(b)) => a == b,
        (Literal::Float(a), Literal::Float(b)) => a.to_bits() == b.to_bits(),
        _ => false,
    }
}

// ── AbstractValue constructors ─────────────────────────────────────

impl AbstractValue {
    pub fn from_literal(lit: &Literal) -> Self {
        match lit {
            Literal::Int(v) => AbstractValue::Finite(FiniteSet::Intervals(SmallVec::from_elem(
                Interval::point(*v),
                1,
            ))),
            Literal::Bool(b) => AbstractValue::Finite(FiniteSet::Bools(SmallVec::from_elem(*b, 1))),
            // Float, Byte, String, List — no arithmetic structure, store as literal.
            Literal::Float(_) | Literal::Byte(_) | Literal::String(_) | Literal::List(_) => {
                AbstractValue::Finite(FiniteSet::Literals(SmallVec::from_elem(lit.clone(), 1)))
            }
            Literal::Unit => AbstractValue::Top,
        }
    }

    pub fn from_known_value(kv: &KnownValue) -> Self {
        match kv {
            KnownValue::Literal(lit) => Self::from_literal(lit),
            KnownValue::Variant { tag, payload } => {
                let payload_val = match payload {
                    Some(p) => Self::from_known_value(p),
                    None => AbstractValue::Bottom,
                };
                AbstractValue::Finite(FiniteSet::Variants(SmallVec::from_elem(
                    (*tag, Box::new(payload_val)),
                    1,
                )))
            }
        }
    }

    pub fn variant(tag: Astr, payload: AbstractValue) -> Self {
        AbstractValue::Finite(FiniteSet::Variants(SmallVec::from_elem(
            (tag, Box::new(payload)),
            1,
        )))
    }

    pub fn tuple(elements: Vec<AbstractValue>) -> Self {
        AbstractValue::Finite(FiniteSet::Tuple(elements))
    }
}

// ── AbstractValue queries (branch pruning) ─────────────────────────

impl AbstractValue {
    /// If this is a definite single boolean, return it.
    pub fn as_definite_bool(&self) -> Option<bool> {
        match self {
            AbstractValue::Finite(FiniteSet::Bools(bs)) if bs.len() == 1 => Some(bs[0]),
            _ => None,
        }
    }

    pub fn tuple_index(&self, index: usize) -> AbstractValue {
        match self {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Finite(FiniteSet::Tuple(elems)) => {
                elems.get(index).cloned().unwrap_or(AbstractValue::Top)
            }
            _ => AbstractValue::Top,
        }
    }

    /// Test whether this value equals a literal.
    /// Returns `{true}`, `{false}`, or `{true,false}` (unknown).
    pub fn test_literal(&self, lit: &Literal) -> AbstractValue {
        match self {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => bool_unknown(),
            AbstractValue::Finite(fs) => match (fs, lit) {
                (FiniteSet::Intervals(ivs), Literal::Int(v)) => {
                    let all_match = ivs.len() == 1 && ivs[0].lo == *v && ivs[0].hi == *v;
                    let any_match = ivs.iter().any(|iv| iv.contains(*v));
                    if all_match {
                        definite_bool(true)
                    } else if !any_match {
                        definite_bool(false)
                    } else {
                        bool_unknown()
                    }
                }
                (FiniteSet::Bools(bs), Literal::Bool(v)) => {
                    if bs.len() == 1 && bs[0] == *v {
                        definite_bool(true)
                    } else if bs.len() == 1 && bs[0] != *v {
                        definite_bool(false)
                    } else {
                        bool_unknown()
                    }
                }
                (FiniteSet::Literals(lits), _) => {
                    let any_match = lits.iter().any(|l| literal_eq(l, lit));
                    let all_match = lits.len() == 1 && any_match;
                    if all_match {
                        definite_bool(true)
                    } else if !any_match {
                        definite_bool(false)
                    } else {
                        bool_unknown()
                    }
                }
                _ => bool_unknown(),
            },
        }
    }

    pub fn test_range(&self, start: i64, end: i64, kind: acvus_ast::RangeKind) -> AbstractValue {
        let (range_lo, range_hi) = effective_range(start, end, kind);

        match self {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => bool_unknown(),
            AbstractValue::Finite(FiniteSet::Intervals(ivs)) => {
                let any_in = ivs.iter().any(|iv| iv.hi >= range_lo && iv.lo <= range_hi);
                let all_in =
                    !ivs.is_empty() && ivs.iter().all(|iv| iv.lo >= range_lo && iv.hi <= range_hi);
                if all_in {
                    definite_bool(true)
                } else if !any_in {
                    definite_bool(false)
                } else {
                    bool_unknown()
                }
            }
            AbstractValue::Finite(FiniteSet::Literals(lits)) => {
                let mut any_in = false;
                let mut any_out = false;
                for lit in lits {
                    if let Literal::Int(v) = lit {
                        if *v >= range_lo && *v <= range_hi {
                            any_in = true;
                        } else {
                            any_out = true;
                        }
                    } else {
                        return bool_unknown();
                    }
                }
                if any_in && !any_out {
                    definite_bool(true)
                } else if !any_in && any_out {
                    definite_bool(false)
                } else {
                    bool_unknown()
                }
            }
            _ => bool_unknown(),
        }
    }

    pub fn test_variant(&self, tag: Astr) -> AbstractValue {
        match self {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => bool_unknown(),
            AbstractValue::Finite(FiniteSet::Variants(vs)) => {
                let has_tag = vs.iter().any(|(t, _)| *t == tag);
                let only_tag = vs.len() == 1 && has_tag;
                if only_tag {
                    definite_bool(true)
                } else if !has_tag {
                    definite_bool(false)
                } else {
                    bool_unknown()
                }
            }
            _ => bool_unknown(),
        }
    }
}

// ── Boolean abstract operations ────────────────────────────────────

fn effective_range(start: i64, end: i64, kind: acvus_ast::RangeKind) -> (i64, i64) {
    match kind {
        acvus_ast::RangeKind::Exclusive => (start, end - 1),
        acvus_ast::RangeKind::InclusiveEnd => (start, end),
        acvus_ast::RangeKind::ExclusiveStart => (start + 1, end),
    }
}

/// A single known boolean value.
fn definite_bool(v: bool) -> AbstractValue {
    AbstractValue::Finite(FiniteSet::Bools(SmallVec::from_elem(v, 1)))
}

/// Unknown boolean — could be true or false.
fn bool_unknown() -> AbstractValue {
    AbstractValue::Finite(FiniteSet::Bools(SmallVec::from_buf([false, true])))
}

pub fn abstract_and(left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    match (left.as_definite_bool(), right.as_definite_bool()) {
        (Some(false), _) | (_, Some(false)) => definite_bool(false),
        (Some(true), Some(true)) => definite_bool(true),
        _ => {
            if matches!(left, AbstractValue::Bottom) || matches!(right, AbstractValue::Bottom) {
                AbstractValue::Bottom
            } else {
                bool_unknown()
            }
        }
    }
}

pub fn abstract_or(left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    match (left.as_definite_bool(), right.as_definite_bool()) {
        (Some(true), _) | (_, Some(true)) => definite_bool(true),
        (Some(false), Some(false)) => definite_bool(false),
        _ => {
            if matches!(left, AbstractValue::Bottom) || matches!(right, AbstractValue::Bottom) {
                AbstractValue::Bottom
            } else {
                bool_unknown()
            }
        }
    }
}

pub fn abstract_not(val: &AbstractValue) -> AbstractValue {
    match val.as_definite_bool() {
        Some(b) => definite_bool(!b),
        None => {
            if matches!(val, AbstractValue::Bottom) {
                AbstractValue::Bottom
            } else {
                bool_unknown()
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_bottom_is_identity() {
        let v = AbstractValue::from_literal(&Literal::Int(42));
        let mut result = AbstractValue::Bottom;
        assert!(result.join_mut(&v));
        assert_eq!(result, v);
    }

    #[test]
    fn join_top_absorbs() {
        let mut v = AbstractValue::from_literal(&Literal::Int(42));
        assert!(v.join_mut(&AbstractValue::Top));
        assert_eq!(v, AbstractValue::Top);
    }

    #[test]
    fn join_same_no_change() {
        let v = AbstractValue::from_literal(&Literal::Int(42));
        let mut result = v.clone();
        assert!(!result.join_mut(&v));
    }

    #[test]
    fn test_literal_exact_match() {
        let v = AbstractValue::from_literal(&Literal::Int(5));
        assert_eq!(
            v.test_literal(&Literal::Int(5)).as_definite_bool(),
            Some(true)
        );
    }

    #[test]
    fn test_literal_no_match() {
        let v = AbstractValue::from_literal(&Literal::Int(3));
        assert_eq!(
            v.test_literal(&Literal::Int(5)).as_definite_bool(),
            Some(false)
        );
    }

    #[test]
    fn test_literal_range_maybe() {
        let v = AbstractValue::Finite(FiniteSet::Intervals(SmallVec::from_elem(
            Interval { lo: 1, hi: 10 },
            1,
        )));
        assert_eq!(v.test_literal(&Literal::Int(5)).as_definite_bool(), None);
    }

    #[test]
    fn test_range_all_in() {
        let v = AbstractValue::from_literal(&Literal::Int(5));
        let result = v.test_range(1, 10, acvus_ast::RangeKind::Exclusive);
        assert_eq!(result.as_definite_bool(), Some(true));
    }

    #[test]
    fn test_range_all_out() {
        let v = AbstractValue::from_literal(&Literal::Int(15));
        let result = v.test_range(1, 10, acvus_ast::RangeKind::Exclusive);
        assert_eq!(result.as_definite_bool(), Some(false));
    }

    #[test]
    fn test_variant_single_match() {
        let tag = unsafe { std::mem::transmute::<u64, Astr>(1u64) };
        let v = AbstractValue::variant(tag, AbstractValue::Bottom);
        assert_eq!(v.test_variant(tag).as_definite_bool(), Some(true));
    }

    #[test]
    fn bool_and_short_circuit() {
        assert_eq!(
            abstract_and(&definite_bool(false), &AbstractValue::Top).as_definite_bool(),
            Some(false)
        );
    }

    #[test]
    fn bool_or_short_circuit() {
        assert_eq!(
            abstract_or(&definite_bool(true), &AbstractValue::Top).as_definite_bool(),
            Some(true)
        );
    }

    #[test]
    fn bool_not_true() {
        assert_eq!(
            abstract_not(&definite_bool(true)).as_definite_bool(),
            Some(false)
        );
    }
}
