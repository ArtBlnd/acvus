use acvus_ast::Literal;
use acvus_utils::Astr;
use smallvec::SmallVec;

use crate::analysis::dataflow::BooleanDomain;
use crate::analysis::reachable_context::KnownValue;

pub trait SemiLattice: Clone + PartialEq {
    fn bottom() -> Self;
    /// Least upper bound. Returns true if self changed.
    fn join_mut(&mut self, other: &Self) -> bool;
}

/// Closed integer interval [lo, hi]. A point value is Interval { lo: n, hi: n }.
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

const MAX_SET_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub enum AbstractValue {
    Bottom,
    Finite(FiniteSet),
    Top,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FiniteSet {
    Intervals(SmallVec<[Interval; 4]>),
    Bools(SmallVec<[bool; 2]>),
    Strings(SmallVec<[Astr; 4]>),
    Variants(SmallVec<[(Astr, Box<AbstractValue>); 4]>),
    Literals(SmallVec<[Literal; 4]>),
    Tuple(Vec<AbstractValue>),
}

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

fn join_finite_sets(a: &FiniteSet, b: &FiniteSet) -> AbstractValue {
    match (a, b) {
        (FiniteSet::Intervals(ai), FiniteSet::Intervals(bi)) => {
            join_intervals(ai, bi)
        }
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
        (FiniteSet::Strings(a_s), FiniteSet::Strings(b_s)) => {
            let mut merged: SmallVec<[Astr; 4]> = a_s.clone();
            for &v in b_s {
                if !merged.contains(&v) {
                    if merged.len() >= MAX_SET_SIZE {
                        return AbstractValue::Top;
                    }
                    merged.push(v);
                }
            }
            AbstractValue::Finite(FiniteSet::Strings(merged))
        }
        (FiniteSet::Variants(av), FiniteSet::Variants(bv)) => {
            join_variants(av, bv)
        }
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
            let mut any_changed = false;
            for (e, other) in elems.iter_mut().zip(bt.iter()) {
                if e.join_mut(other) {
                    any_changed = true;
                }
            }
            let _ = any_changed;
            AbstractValue::Finite(FiniteSet::Tuple(elems))
        }
        _ => AbstractValue::Top,
    }
}

fn join_intervals(a: &[Interval], b: &[Interval]) -> AbstractValue {
    let mut all: SmallVec<[Interval; 8]> = SmallVec::new();
    all.extend_from_slice(a);
    all.extend_from_slice(b);
    all.sort_by(|x, y| x.lo.cmp(&y.lo).then(x.hi.cmp(&y.hi)));

    // Merge overlapping/adjacent
    let mut merged: SmallVec<[Interval; 4]> = SmallVec::new();
    for iv in all {
        if let Some(last) = merged.last_mut() {
            if last.overlaps_or_adjacent(&iv) {
                *last = last.merge(&iv);
                continue;
            }
        }
        merged.push(iv);
    }

    // Graduated widening: merge closest pairs until within limit
    while merged.len() > MAX_SET_SIZE {
        if merged.len() < 2 {
            break;
        }
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

    if merged.len() > MAX_SET_SIZE {
        AbstractValue::Top
    } else {
        AbstractValue::Finite(FiniteSet::Intervals(merged))
    }
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

impl BooleanDomain for AbstractValue {
    fn as_definite_bool(&self) -> Option<bool> {
        match self {
            AbstractValue::Finite(FiniteSet::Bools(bs)) if bs.len() == 1 => Some(bs[0]),
            _ => None,
        }
    }
}

impl AbstractValue {
    pub fn from_literal(lit: &Literal) -> Self {
        match lit {
            Literal::Int(v) => {
                AbstractValue::Finite(FiniteSet::Intervals(SmallVec::from_elem(
                    Interval::point(*v),
                    1,
                )))
            }
            Literal::Bool(b) => {
                AbstractValue::Finite(FiniteSet::Bools(SmallVec::from_elem(*b, 1)))
            }
            Literal::Float(_) | Literal::Byte(_) | Literal::List(_) => {
                AbstractValue::Finite(FiniteSet::Literals(SmallVec::from_elem(
                    lit.clone(),
                    1,
                )))
            }
            Literal::String(_) => {
                // String literals in Literal use String, not Astr.
                // We store them in the Literals variant.
                AbstractValue::Finite(FiniteSet::Literals(SmallVec::from_elem(
                    lit.clone(),
                    1,
                )))
            }
        }
    }

    pub fn from_known_value(kv: &KnownValue) -> Self {
        match kv {
            KnownValue::Literal(lit) => Self::from_literal(lit),
            KnownValue::Variant { tag, payload } => {
                let payload_val = match payload {
                    Some(p) => Self::from_known_value(p),
                    None => AbstractValue::Finite(FiniteSet::Literals(SmallVec::new())),
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

    pub fn tuple_index(&self, index: usize) -> AbstractValue {
        match self {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Finite(FiniteSet::Tuple(elems)) => {
                elems.get(index).cloned().unwrap_or(AbstractValue::Top)
            }
            _ => AbstractValue::Top,
        }
    }

    pub fn test_literal(&self, lit: &Literal) -> AbstractValue {
        match self {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => bool_top(),
            AbstractValue::Finite(fs) => match (fs, lit) {
                (FiniteSet::Intervals(ivs), Literal::Int(v)) => {
                    let any_match = ivs.iter().any(|iv| iv.contains(*v));
                    let all_match = ivs.len() == 1
                        && ivs[0].lo == *v
                        && ivs[0].hi == *v;
                    if all_match {
                        bool_single(true)
                    } else if !any_match {
                        bool_single(false)
                    } else {
                        bool_top()
                    }
                }
                (FiniteSet::Bools(bs), Literal::Bool(v)) => {
                    if bs.len() == 1 && bs[0] == *v {
                        bool_single(true)
                    } else if bs.len() == 1 && bs[0] != *v {
                        bool_single(false)
                    } else {
                        bool_top()
                    }
                }
                (FiniteSet::Strings(ss), Literal::String(v)) => {
                    // Astr vs String comparison not possible without interner.
                    // Fall back to Top.
                    let _ = (ss, v);
                    bool_top()
                }
                (FiniteSet::Literals(lits), _) => {
                    let any_match = lits.iter().any(|l| literal_eq(l, lit));
                    let all_match = lits.len() == 1 && any_match;
                    if all_match {
                        bool_single(true)
                    } else if !any_match {
                        bool_single(false)
                    } else {
                        bool_top()
                    }
                }
                _ => bool_top(),
            },
        }
    }

    pub fn test_range(
        &self,
        start: i64,
        end: i64,
        kind: acvus_ast::RangeKind,
    ) -> AbstractValue {
        match self {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => bool_top(),
            AbstractValue::Finite(FiniteSet::Intervals(ivs)) => {
                let (range_lo, range_hi) = effective_range(start, end, kind);
                let any_in = ivs.iter().any(|iv| iv.hi >= range_lo && iv.lo <= range_hi);
                let all_in = ivs
                    .iter()
                    .all(|iv| iv.lo >= range_lo && iv.hi <= range_hi);
                if all_in && !ivs.is_empty() {
                    bool_single(true)
                } else if !any_in {
                    bool_single(false)
                } else {
                    bool_top()
                }
            }
            AbstractValue::Finite(FiniteSet::Literals(lits)) => {
                let (range_lo, range_hi) = effective_range(start, end, kind);
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
                        return bool_top();
                    }
                }
                if any_in && !any_out {
                    bool_single(true)
                } else if !any_in && any_out {
                    bool_single(false)
                } else {
                    bool_top()
                }
            }
            _ => bool_top(),
        }
    }

    pub fn test_variant(&self, tag: Astr) -> AbstractValue {
        match self {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => bool_top(),
            AbstractValue::Finite(FiniteSet::Variants(vs)) => {
                let has_tag = vs.iter().any(|(t, _)| *t == tag);
                let only_tag = vs.len() == 1 && has_tag;
                if only_tag {
                    bool_single(true)
                } else if !has_tag {
                    bool_single(false)
                } else {
                    bool_top()
                }
            }
            _ => bool_top(),
        }
    }
}

fn effective_range(start: i64, end: i64, kind: acvus_ast::RangeKind) -> (i64, i64) {
    match kind {
        acvus_ast::RangeKind::Exclusive => (start, end - 1),
        acvus_ast::RangeKind::InclusiveEnd => (start, end),
        acvus_ast::RangeKind::ExclusiveStart => (start + 1, end),
    }
}

fn bool_single(v: bool) -> AbstractValue {
    AbstractValue::Finite(FiniteSet::Bools(SmallVec::from_elem(v, 1)))
}

fn bool_top() -> AbstractValue {
    let mut bs = SmallVec::new();
    bs.push(false);
    bs.push(true);
    AbstractValue::Finite(FiniteSet::Bools(bs))
}

pub fn abstract_and(left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    match (left.as_definite_bool(), right.as_definite_bool()) {
        (Some(false), _) | (_, Some(false)) => bool_single(false),
        (Some(true), Some(true)) => bool_single(true),
        _ => {
            if matches!(left, AbstractValue::Bottom) || matches!(right, AbstractValue::Bottom) {
                AbstractValue::Bottom
            } else {
                bool_top()
            }
        }
    }
}

pub fn abstract_or(left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    match (left.as_definite_bool(), right.as_definite_bool()) {
        (Some(true), _) | (_, Some(true)) => bool_single(true),
        (Some(false), Some(false)) => bool_single(false),
        _ => {
            if matches!(left, AbstractValue::Bottom) || matches!(right, AbstractValue::Bottom) {
                AbstractValue::Bottom
            } else {
                bool_top()
            }
        }
    }
}

pub fn abstract_not(val: &AbstractValue) -> AbstractValue {
    match val.as_definite_bool() {
        Some(b) => bool_single(!b),
        None => {
            if matches!(val, AbstractValue::Bottom) {
                AbstractValue::Bottom
            } else {
                bool_top()
            }
        }
    }
}

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
        assert_eq!(v.test_literal(&Literal::Int(5)).as_definite_bool(), Some(true));
    }

    #[test]
    fn test_literal_no_match() {
        let v = AbstractValue::from_literal(&Literal::Int(3));
        assert_eq!(v.test_literal(&Literal::Int(5)).as_definite_bool(), Some(false));
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
            abstract_and(&bool_single(false), &AbstractValue::Top)
                .as_definite_bool(),
            Some(false)
        );
    }

    #[test]
    fn bool_or_short_circuit() {
        assert_eq!(
            abstract_or(&bool_single(true), &AbstractValue::Top)
                .as_definite_bool(),
            Some(true)
        );
    }

    #[test]
    fn bool_not_true() {
        assert_eq!(abstract_not(&bool_single(true)).as_definite_bool(), Some(false));
    }
}
