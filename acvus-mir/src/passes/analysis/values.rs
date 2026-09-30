//! The value analyses: the sparse engine over the constant, the interval
//! and the affine domains (passes-contracts: `sccp` reads `ConstantValues`,
//! `bce` reads `IntervalValues` and `AffineValues` as their reduced
//! product, `ValueBounds`). A value operation is its domain's; every other
//! instruction is `top`, but that a call's result lies where its callee's
//! declared Relations put it (extern-fn §3), as the affine domain reads
//! them.

use acvus_ast::Literal;

use crate::decl::Declarations;
use crate::decl::declared::{Atom, Before, Subject};
use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Definition, Inst, Module, Op, Terminator};
use crate::ir::{Callee, ExternImpl, ForSource, SwitchKey, ValueId, known_len};
use crate::passes::analysis::{Analysis, AnalysisManager, Executable, SparseAnalysis, SparseFacts, solve_sparse};
use crate::passes::domain::{
    Affine, AffineDomain, Bound, Combination, Constant, ConstantDomain, Interval, IntervalDomain, Limit, Symbol, ValueDomain,
    bounds_on,
};
use crate::ty::{CastTy, Concrete, TyKind, Types};

/// Every value's constant, conditional on the arms a known condition takes
/// (Wegman and Zadeck). Cost: the sparse engine's, with height 3.
pub struct ConstantValues;

/// Every integer value's interval, every arm taken. Cost: the sparse
/// engine's, a parameter of a loop header rising at most three times under
/// widening.
pub struct IntervalValues;

/// Every integer value's affine combination of symbols, every arm taken:
/// a slice's length is its `Len`, a `For`'s counter is its end less one
/// less its `Remaining` steps, and a call's result is what its declared
/// Relations make it equal. Cost: the sparse engine's, with height 3.
pub struct AffineValues;

impl Analysis for ConstantValues {
    type Facts = SparseFacts<ConstantDomain>;

    const READS: &'static [Aspect] = &[Aspect::Instructions, Aspect::CfgShape, Aspect::Types];

    fn compute(_am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts {
        let body = module.body(body);
        solve_sparse(&Constants { operand_types: OperandTypes { types: module.types(), body } }, body)
    }
}

impl Analysis for IntervalValues {
    type Facts = SparseFacts<IntervalDomain>;

    const READS: &'static [Aspect] = &[Aspect::Instructions, Aspect::CfgShape, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts {
        let body = module.body(body);
        let intervals = Intervals { operand_types: OperandTypes { types: module.types(), body }, decls: am.decls() };
        solve_sparse(&intervals, body)
    }
}

impl Analysis for AffineValues {
    type Facts = SparseFacts<AffineDomain>;

    const READS: &'static [Aspect] = &[Aspect::Instructions, Aspect::CfgShape, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts {
        let body = module.body(body);
        let affines = Affines { operand_types: OperandTypes { types: module.types(), body }, decls: am.decls() };
        solve_sparse(&affines, body)
    }
}

impl SparseFacts<ConstantDomain> {
    /// `value`'s one value, where it is known.
    pub fn constant(&self, value: ValueId) -> Option<&Literal> {
        match self.of(value) {
            Constant::Known(literal) => Some(literal),
            Constant::Bottom | Constant::Top => None,
        }
    }
}

/// The type a value operation computes at, read from the body's values.
pub(super) struct OperandTypes<'m> {
    pub(super) types: &'m Types<Concrete>,
    pub(super) body: &'m Body,
}

impl OperandTypes<'_> {
    /// `value`'s type, where it is one a value operation computes at.
    fn of(&self, value: ValueId) -> Option<CastTy> {
        CastTy::of_kind(self.types.kind(self.body.ty_of(value)))
    }

    fn is_integer(&self, value: ValueId) -> bool {
        matches!(self.of(value), Some(CastTy::Int(_)))
    }

    /// Whether `value` is a reference to a slice, whose element count no
    /// run changes while it exists.
    fn is_slice(&self, value: ValueId) -> bool {
        match self.types.kind(self.body.ty_of(value)) {
            TyKind::Ref(_, lent) => matches!(self.types.kind(*lent), TyKind::Slice(_)),
            _ => false,
        }
    }

    /// The domain's value of a value operation, or `None` where `inst` is
    /// none or its operands have no scalar type (D60).
    pub(super) fn value_operation<D>(&self, domain: &D, inst: &Inst, operands: &[&D::Value]) -> Option<D::Value>
    where
        D: ValueDomain,
    {
        Some(match (inst.op(), operands) {
            (Op::Const { value }, []) => {
                let result = inst.result().expect("a constant yields a value");
                domain.constant(value, self.of(result)?)
            }
            (Op::BinOp { op, left, .. }, [left_fact, right_fact]) => {
                domain.binary(*op, self.of(*left)?, left_fact, right_fact)
            }
            (Op::UnaryOp { op, operand }, [operand_fact]) => domain.unary(*op, self.of(*operand)?, operand_fact),
            (Op::Cast { src, to }, [operand_fact]) => domain.cast(self.of(*src)?, *to, operand_fact),
            (Op::Const { .. } | Op::BinOp { .. } | Op::UnaryOp { .. } | Op::Cast { .. }, _) => {
                panic!("{:?} is handed {} operand facts, not one per operand", inst.op(), operands.len())
            }
            _ => return None,
        })
    }
}

struct Constants<'m> {
    operand_types: OperandTypes<'m>,
}

impl SparseAnalysis for Constants<'_> {
    type Domain = ConstantDomain;

    fn domain(&self) -> &ConstantDomain {
        &ConstantDomain
    }

    fn transfer(&self, inst: &Inst, operands: &[&Constant]) -> Constant {
        self.operand_types.value_operation(&ConstantDomain, inst, operands).unwrap_or(Constant::Top)
    }

    /// Only the arm a known `bool` condition or a known switch key chooses
    /// runs; no arm runs yet while the condition is `bottom`. A string key
    /// is compared through the interner, which the module does not hold,
    /// so a string switch takes every arm (D60).
    fn executable(&self, terminator: &Terminator, operands: &[&Constant]) -> Executable {
        match terminator {
            Terminator::JumpIf { then_to, else_to, .. } | Terminator::Diamond { then_to, else_to, .. } => {
                branch(operands[0], *then_to, *else_to)
            }
            Terminator::While { stages, exit, .. } => branch(operands[0], stages.body(), *exit),
            Terminator::Switch { arms, default, .. } => {
                let known = match operands[0] {
                    Constant::Bottom => return Executable::Only(Vec::new()),
                    Constant::Top => return Executable::All,
                    Constant::Known(literal) => literal.desugared(),
                };
                let Some(key) = switch_key(&known) else {
                    return Executable::All;
                };
                let chosen = arms
                    .iter()
                    .find(|(arm, _, _)| *arm == key)
                    .map(|(_, to, _)| *to)
                    .or(default.as_ref().map(|(to, _)| *to));
                Executable::Only(chosen.into_iter().collect())
            }
            Terminator::Jump { .. } | Terminator::For { .. } | Terminator::Return { .. } | Terminator::Diverge | Terminator::NoArmMatched { .. } => {
                Executable::All
            }
        }
    }

    /// A `For`'s counter and element are `top`; its trip count is known
    /// where both ends of its range are.
    fn supplied(&self, terminator: &Terminator, to: BlockId, _index: usize, operands: &[&Constant]) -> Constant {
        let Terminator::For { source, exit, .. } = terminator else {
            panic!("only a For supplies parameters, not {terminator:?}")
        };
        if to != *exit {
            return Constant::Top;
        }
        match (source, operands) {
            (ForSource::Range { .. }, [Constant::Known(at), Constant::Known(hi), ..]) => {
                let (Literal::Int(at), Literal::Int(hi)) = (at.desugared(), hi.desugared()) else {
                    panic!("a range's ends are integers, not {at:?} and {hi:?}")
                };
                Constant::Known(Literal::Int((hi - at).max(0)))
            }
            _ => Constant::Top,
        }
    }
}

/// The switch key a known value is, where it is one a switch narrows on.
pub(in crate::passes) fn switch_key(known: &Literal) -> Option<SwitchKey> {
    match known {
        Literal::Int(n) => Some(SwitchKey::Int(*n)),
        Literal::Bool(b) => Some(SwitchKey::Bool(*b)),
        Literal::Char(c) => Some(SwitchKey::Char(*c)),
        Literal::String(_) => None,
        Literal::Float(_) | Literal::Unit | Literal::List(_) | Literal::IntOf(_) | Literal::Bytes(_) => {
            panic!("a switch is not on {known:?}")
        }
    }
}

/// On `bottom` no arm runs yet, on a known truth only its arm, else both.
fn branch(condition: &Constant, on_true: BlockId, on_false: BlockId) -> Executable {
    match condition {
        Constant::Bottom => Executable::Only(Vec::new()),
        Constant::Top => Executable::All,
        Constant::Known(Literal::Bool(true)) => Executable::Only(vec![on_true]),
        Constant::Known(Literal::Bool(false)) => Executable::Only(vec![on_false]),
        Constant::Known(other) => panic!("a branch's condition is a bool, not {other:?}"),
    }
}

struct Intervals<'m> {
    operand_types: OperandTypes<'m>,
    decls: &'m Declarations,
}

impl SparseAnalysis for Intervals<'_> {
    type Domain = IntervalDomain;

    fn domain(&self) -> &IntervalDomain {
        &IntervalDomain
    }

    fn transfer(&self, inst: &Inst, operands: &[&Interval]) -> Interval {
        if let Some(value) = self.operand_types.value_operation(&IntervalDomain, inst, operands) {
            return value;
        }
        match inst.op() {
            Op::FunctionCall { callee: Callee::Extern { id, r#impl, .. }, args, .. }
                if inst.result().is_some_and(|result| self.operand_types.is_integer(result)) =>
            {
                assert_eq!(args.len(), operands.len(), "an extern call's operands are its arguments");
                let relations = &self.decls.call_facts(*id, *r#impl).relations;
                let arg_interval = |atom: &Atom| declared_atom_interval(&self.operand_types, args, operands, atom);
                bounds_on(relations, Atom::Result)
                    .iter()
                    .filter_map(|limit| limit_interval(limit, arg_interval))
                    .fold(Interval::full(), |result, range| result.meet(&range))
            }
            _ => Interval::full(),
        }
    }

    fn executable(&self, _terminator: &Terminator, _operands: &[&Interval]) -> Executable {
        Executable::All
    }

    /// A range's counter lies in `[at, hi - 1]` and its trip count is
    /// `max(hi - at, 0)`; a slice's or an array's counter and trip count
    /// are counts, and its element is `top` (D60).
    fn supplied(&self, terminator: &Terminator, to: BlockId, index: usize, operands: &[&Interval]) -> Interval {
        let Terminator::For { source, stages, .. } = terminator else {
            panic!("only a For supplies parameters, not {terminator:?}")
        };
        let into_body = to == stages.body();
        match (source, operands) {
            (ForSource::Range { .. }, [at, hi, ..]) => match into_body {
                true => at.at_least_lo().meet(&hi.difference(&Interval::exactly(1)).at_most_hi()),
                false => hi.difference(at).greatest(&Interval::exactly(0)),
            },
            (ForSource::Range { .. }, _) => panic!("a range For reads its two ends"),
            (ForSource::Slice(_) | ForSource::SliceMut(_) | ForSource::Array(_), _) => {
                match into_body && index != source.counter_param() {
                    true => Interval::full(),
                    false => Interval::exactly(0).at_least_lo(),
                }
            }
        }
    }
}

/// The interval a declared Relation's atom lies in, over one call's
/// arguments: an integer argument's own, and a count for a length; `None`
/// for an atom the call's arguments do not bound, such as the result or
/// what a reference lends.
fn declared_atom_interval(types: &OperandTypes<'_>, args: &[ValueId], intervals: &[&Interval], atom: &Atom) -> Option<Interval> {
    match atom {
        Atom::Arg(at) if types.is_integer(args[*at]) => Some(intervals[*at].clone()),
        Atom::Len(Subject::Arg(_)) | Atom::Old(Before::Len(_)) => Some(Interval::exactly(0).at_least_lo()),
        Atom::Arg(_) | Atom::Result | Atom::Len(Subject::Result) | Atom::Old(Before::Lent(_)) => None,
    }
}

/// The interval a combination lies in where each atom lies in the interval
/// `atom` gives it; `None` where `atom` gives none for one of them.
fn combination_interval<A, F>(combination: &Combination<A>, mut atom: F) -> Option<Interval>
where
    A: Ord + Clone,
    F: FnMut(&A) -> Option<Interval>,
{
    let mut out = Interval::exactly(combination.constant_term());
    for (at, coefficient) in combination.terms() {
        out = out.sum(&Interval::exactly(coefficient).product(&atom(at)?));
    }
    Some(out)
}

/// The interval a declared limit puts its atom in.
fn limit_interval<A, F>(limit: &Limit<A>, atom: F) -> Option<Interval>
where
    A: Ord + Clone,
    F: FnMut(&A) -> Option<Interval>,
{
    Some(match limit {
        Limit::Exactly(at) => combination_interval(at, atom)?,
        Limit::AtMost(at) => combination_interval(at, atom)?.at_most_hi(),
        Limit::AtLeast(at) => combination_interval(at, atom)?.at_least_lo(),
    })
}

/// A value's combination, and where none is known, the value itself, which
/// every value equals; `None` where no run defines it.
fn named(value: ValueId, fact: &Affine) -> Option<Combination<Symbol>> {
    match fact {
        Affine::Top => Some(Combination::atom(Symbol::Value(value))),
        Affine::Exactly(combination) => Some(combination.clone()),
        Affine::Bottom => None,
    }
}

struct Affines<'m> {
    operand_types: OperandTypes<'m>,
    decls: &'m Declarations,
}

impl Affines<'_> {
    /// The combination a call's declared Relations make its integer result
    /// equal, over its arguments' combinations; `top` where none does.
    fn declared_result(
        &self,
        inst: &Inst,
        id: acvus_utils::QualifiedRef,
        r#impl: Option<usize>,
        args: &[ValueId],
        named: &[Option<Combination<Symbol>>],
    ) -> Affine {
        let types = &self.operand_types;
        if !inst.result().is_some_and(|result| types.is_integer(result)) {
            return Affine::Top;
        }
        let relations = &self.decls.call_facts(id, r#impl).relations;
        let atom = |atom: &Atom| match atom {
            Atom::Arg(at) if types.is_integer(args[*at]) => named[*at].clone(),
            Atom::Len(Subject::Arg(at)) if types.is_slice(args[*at]) => Some(Combination::atom(Symbol::Len(args[*at]))),
            Atom::Arg(_) | Atom::Len(_) | Atom::Result | Atom::Old(_) => None,
        };
        bounds_on(relations, Atom::Result)
            .iter()
            .find_map(|limit| match limit.substitute(atom)? {
                Limit::Exactly(combination) => Some(Affine::Exactly(combination)),
                Limit::AtMost(_) | Limit::AtLeast(_) => None,
            })
            .unwrap_or(Affine::Top)
    }
}

impl SparseAnalysis for Affines<'_> {
    type Domain = AffineDomain;

    fn domain(&self) -> &AffineDomain {
        &AffineDomain
    }

    /// An operand of no known combination is read as itself.
    fn transfer(&self, inst: &Inst, operands: &[&Affine]) -> Affine {
        let body = self.operand_types.body;
        let ids: Vec<ValueId> = inst.operands().into_iter().filter(|value| !body.is_token(*value)).collect();
        assert_eq!(ids.len(), operands.len(), "one fact per operand that is not an order token");
        let named: Vec<Option<Combination<Symbol>>> = ids.iter().zip(operands).map(|(id, fact)| named(*id, fact)).collect();
        let named_facts: Vec<Affine> = named.iter().map(|named| named.clone().map_or(Affine::Bottom, Affine::Exactly)).collect();
        let named_facts: Vec<&Affine> = named_facts.iter().collect();
        if let Some(value) = self.operand_types.value_operation(&AffineDomain, inst, &named_facts) {
            return value;
        }
        match inst.op() {
            Op::SliceLen { src } => Affine::Exactly(Combination::atom(Symbol::Len(*src))),
            Op::FunctionCall { callee: Callee::Extern { id, r#impl, .. }, args, .. } => {
                self.declared_result(inst, *id, *r#impl, args, &named)
            }
            _ => Affine::Top,
        }
    }

    fn executable(&self, _terminator: &Terminator, _operands: &[&Affine]) -> Executable {
        Executable::All
    }

    /// A counter entering a `For`'s body is its end less one less the steps
    /// left, a range's end its `hi` and a slice's its length. An array's
    /// counter, an element and a trip count are `top`.
    fn supplied(&self, terminator: &Terminator, to: BlockId, index: usize, operands: &[&Affine]) -> Affine {
        let Terminator::For { source, stages, .. } = terminator else {
            panic!("only a For supplies parameters, not {terminator:?}")
        };
        if to != stages.body() || index != source.counter_param() {
            return Affine::Top;
        }
        let end = match (source, operands) {
            (ForSource::Range { hi, .. }, [_, hi_fact, ..]) => match named(*hi, hi_fact) {
                Some(end) => end,
                None => return Affine::Bottom,
            },
            (ForSource::Range { .. }, _) => panic!("a range For reads its two ends"),
            (ForSource::Slice(slice) | ForSource::SliceMut(slice), _) => Combination::atom(Symbol::Len(*slice)),
            (ForSource::Array(_), _) => return Affine::Top,
        };
        let counter = self.operand_types.body.block(to).params()[index];
        let steps_after = Combination::constant(1).checked_add(&Combination::atom(Symbol::Remaining(counter)));
        match steps_after.and_then(|steps_after| end.checked_sub(&steps_after)) {
            Some(counter) => Affine::Exactly(counter),
            None => Affine::Top,
        }
    }
}

/// The reduced product of one body's interval and affine facts (D83): a
/// value lies in its interval and in the interval its combination lies in,
/// each symbol in its own; a slice's length is what its maker's declared
/// Relations or its array's type make it. This is how `i < len(s)` is
/// proven for a length no interval bounds.
pub(in crate::passes) struct ValueBounds<'f> {
    pub intervals: &'f SparseFacts<IntervalDomain>,
    pub affine: &'f SparseFacts<AffineDomain>,
    pub decls: &'f Declarations,
    pub body: &'f Body,
    pub types: &'f Types<Concrete>,
}

/// The extern impl that made a slice, and its arguments: a call's, or an
/// `AsSlice`'s, whose one argument is its container.
struct MadeBy<'b> {
    r#impl: ExternImpl,
    args: &'b [ValueId],
}

impl ValueBounds<'_> {
    /// `value`'s combination; `None` where no run defines it.
    fn combination(&self, value: ValueId) -> Option<Combination<Symbol>> {
        named(value, self.affine.of(value))
    }

    fn symbol_interval(&self, symbol: &Symbol) -> Interval {
        match symbol {
            Symbol::Value(value) => self.intervals.of(*value).clone(),
            Symbol::Len(slice) => self.len_interval(*slice),
            Symbol::Remaining(_) => Interval::exactly(0).at_least_lo(),
        }
    }

    fn interval_of(&self, combination: &Combination<Symbol>) -> Interval {
        combination_interval(combination, |symbol| Some(self.symbol_interval(symbol)))
            .expect("every symbol lies in an interval")
    }

    /// The integers `value` may be on a run that defines it. Cost: O(the
    /// symbols of its combination · the relations of a slice's impl).
    pub fn value_interval(&self, value: ValueId) -> Interval {
        match self.combination(value) {
            Some(combination) => self.intervals.of(value).meet(&self.interval_of(&combination)),
            None => Interval::Empty,
        }
    }

    /// Whether `0 ≤ index < len(slice)` on every run that reaches an index
    /// of `slice` at `index`: the index's interval lies below the least
    /// length, or the length less the index less one, as a combination,
    /// is never negative. A slice whose relations admit no length is made
    /// by no run. Cost: as `value_interval`, for both.
    pub fn index_in_bounds(&self, slice: ValueId, index: ValueId) -> bool {
        let (at, len) = (self.value_interval(index), self.len_interval(slice));
        let (Interval::Range { lo, hi }, Interval::Range { lo: shortest, .. }) = (&at, &len) else {
            return true;
        };
        if *lo < Bound::Finite(0) {
            return false;
        }
        if hi < shortest {
            return true;
        }
        let Some(index) = self.combination(index) else {
            return true;
        };
        let one = Combination::constant(1);
        let slack = self.len_combination(slice).checked_sub(&index).and_then(|slack| slack.checked_sub(&one));
        slack.is_some_and(|slack| match self.interval_of(&slack) {
            Interval::Empty => true,
            Interval::Range { lo, .. } => lo >= Bound::Finite(0),
        })
    }

    fn made_by(&self, slice: ValueId) -> Option<MadeBy<'_>> {
        let Definition::Inst(made) = self.body.definition(slice) else {
            return None;
        };
        match self.body.inst(made).op() {
            Op::FunctionCall { callee: Callee::Extern { id, r#impl, .. }, args, .. } => {
                Some(MadeBy { r#impl: ExternImpl { id: *id, r#impl: *r#impl }, args })
            }
            Op::AsSlice { container, r#impl, .. } => Some(MadeBy { r#impl: *r#impl, args: std::slice::from_ref(container) }),
            _ => None,
        }
    }

    /// The limits its maker's declared Relations put on a slice's length,
    /// over its maker's arguments.
    fn declared_len_limits(&self, made_by: &MadeBy<'_>) -> Vec<Limit<Atom>> {
        let facts = self.decls.call_facts(made_by.r#impl.id, made_by.r#impl.r#impl);
        bounds_on(&facts.relations, Atom::Len(Subject::Result))
    }

    /// The interval a slice's length lies in: a count, within what its
    /// maker's declared Relations put it in over its maker's arguments'
    /// intervals, and exactly the length its array's type carries.
    fn len_interval(&self, slice: ValueId) -> Interval {
        let mut len = Interval::exactly(0).at_least_lo();
        if let Some(made_by) = self.made_by(slice) {
            let operand_types = OperandTypes { types: self.types, body: self.body };
            let intervals: Vec<&Interval> = made_by.args.iter().map(|arg| self.intervals.of(*arg)).collect();
            let arg_interval = |atom: &Atom| declared_atom_interval(&operand_types, made_by.args, &intervals, atom);
            for limit in self.declared_len_limits(&made_by) {
                if let Some(range) = limit_interval(&limit, arg_interval) {
                    len = len.meet(&range);
                }
            }
        }
        if let Some(carried) = self.carried_len(slice) {
            len = len.meet(&Interval::exactly(carried));
        }
        len
    }

    /// A slice's length as a combination: what its maker's declared
    /// Relations make it equal over its maker's arguments' combinations,
    /// the length its array's type carries, or its own `Len`.
    fn len_combination(&self, slice: ValueId) -> Combination<Symbol> {
        let operand_types = OperandTypes { types: self.types, body: self.body };
        let declared = self.made_by(slice).and_then(|made_by| {
            let atom = |atom: &Atom| match atom {
                Atom::Arg(at) if operand_types.is_integer(made_by.args[*at]) => self.combination(made_by.args[*at]),
                Atom::Len(Subject::Arg(at)) if operand_types.is_slice(made_by.args[*at]) => {
                    Some(Combination::atom(Symbol::Len(made_by.args[*at])))
                }
                Atom::Arg(_) | Atom::Len(_) | Atom::Result | Atom::Old(_) => None,
            };
            self.declared_len_limits(&made_by).iter().find_map(|limit| match limit.substitute(atom)? {
                Limit::Exactly(len) => Some(len),
                Limit::AtMost(_) | Limit::AtLeast(_) => None,
            })
        });
        declared
            .or_else(|| self.carried_len(slice).map(Combination::constant))
            .unwrap_or_else(|| Combination::atom(Symbol::Len(slice)))
    }

    /// The length a slice's type carries: an `AsSlice` of a reference to an
    /// `Array<T, N>` has `N` elements. The length of another container is
    /// not read (D66).
    fn carried_len(&self, slice: ValueId) -> Option<i128> {
        let Definition::Inst(made) = self.body.definition(slice) else {
            return None;
        };
        let Op::AsSlice { container, .. } = self.body.inst(made).op() else {
            return None;
        };
        let TyKind::Ref(_, lent) = self.types.kind(self.body.ty_of(*container)) else {
            panic!("an AsSlice's container {container:?} is a reference")
        };
        match self.types.kind(*lent) {
            TyKind::Array(_, len, _) => Some(i128::try_from(known_len(len)).expect("an array's length fits an i128")),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;
    use acvus_utils::{Interner, QualifiedRef};

    use super::*;
    use crate::decl::declared::{Comparison, Relation, Term};
    use crate::decl::{Core, DeclaredFacts, GenericImplSig, Impls, LoopMarkers, RawDeclarations, Scheme};
    use crate::ir::cfg::Stages;
    use crate::ir::{BinOp, ExitTrip, Overflow};
    use crate::passes::domain::Bound;
    use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Poly, Ty, TyId, TyKind};

    struct Fixture {
        module: Module,
        interner: Interner,
        int: Ty,
        byte: Ty,
        bool: Ty,
    }

    fn fixture() -> Fixture {
        let mut types = Types::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let byte = types.intern(TyKind::Int(IntTy::I8));
        let bool = types.intern(TyKind::Bool);
        Fixture { module: Module::new(types), interner: Interner::new(), int, byte, bool }
    }

    fn declarations(interner: &Interner, functions: Vec<(QualifiedRef, Scheme)>, types: Types<Poly>) -> Declarations {
        Declarations::verify(RawDeclarations {
            types,
            functions,
            lang_impls: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            entries: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(interner),
        })
        .expect("the declarations verify")
    }

    fn constant(n: i128) -> Op {
        Op::Const { value: Literal::Int(n) }
    }

    fn binary(op: BinOp, left: ValueId, right: ValueId) -> Op {
        Op::BinOp { op, left, right }
    }

    fn jump(to: BlockId, args: Vec<ValueId>) -> Terminator {
        Terminator::Jump { to, args }
    }

    fn ret(value: ValueId) -> Terminator {
        Terminator::Return { value, order: None, span: Span::ZERO }
    }

    fn finite(lo: i128, hi: i128) -> Interval {
        Interval::Range { lo: Bound::Finite(lo), hi: Bound::Finite(hi) }
    }

    fn facts<A>(module: &Module, decls: &Declarations, id: BodyId) -> std::rc::Rc<A::Facts>
    where
        A: Analysis,
    {
        AnalysisManager::new(decls).get::<A>(module, id)
    }

    /// `x = 2; y = 3; z = x + y; w = z * a; b = 100i8; o = b + b; r = z as i8`:
    /// the constants fold, a parameter is `top`, and an `i8` overflow is a
    /// trap, so `o` is no constant and no value above `i8::MAX`.
    #[test]
    fn a_straight_line_folds() {
        let mut f = fixture();
        let decls = declarations(&f.interner, Vec::new(), Types::new());
        let (name, int, byte) = (f.interner.intern("a"), f.int, f.byte);
        let mut builder = f.module.build_body(false, int);
        let a = builder.param(name, int, acvus_ast::Span::ZERO);
        let x = builder.push(constant(2), int, Span::ZERO);
        let y = builder.push(constant(3), int, Span::ZERO);
        let z = builder.push(binary(BinOp::Add(Overflow::Trap), x, y), int, Span::ZERO);
        let w = builder.push(binary(BinOp::Mul(Overflow::Trap), z, a), int, Span::ZERO);
        let b = builder.push(constant(100), byte, Span::ZERO);
        let o = builder.push(binary(BinOp::Add(Overflow::Trap), b, b), byte, Span::ZERO);
        let r = builder.push(Op::Cast { src: z, to: CastTy::Int(IntTy::I8) }, byte, Span::ZERO);
        builder.terminate(ret(w), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let constants = facts::<ConstantValues>(&f.module, &decls, id);
        assert_eq!(constants.constant(z), Some(&Literal::Int(5)));
        assert_eq!(constants.constant(r), Some(&Literal::Int(5)));
        assert_eq!(*constants.of(a), Constant::Top);
        assert_eq!(*constants.of(w), Constant::Top);
        assert_eq!(*constants.of(o), Constant::Top, "100 + 100 traps at i8");

        let intervals = facts::<IntervalValues>(&f.module, &decls, id);
        assert_eq!(*intervals.of(z), finite(5, 5));
        assert_eq!(*intervals.of(o), Interval::Empty, "no run passes 100 + 100 at i8");
        assert_eq!(*intervals.of(w), finite(i128::from(i64::MIN), i128::from(i64::MAX)));
    }

    /// `c = 1 < 2; if c { t = 10 } else { e = 20 }; join(p)`: the constants
    /// run only the then arm, the intervals run both.
    #[test]
    fn a_diamond_on_a_known_condition_runs_one_arm() {
        let mut f = fixture();
        let decls = declarations(&f.interner, Vec::new(), Types::new());
        let (int, bool) = (f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let one = builder.push(constant(1), int, Span::ZERO);
        let two = builder.push(constant(2), int, Span::ZERO);
        let c = builder.push(binary(BinOp::Lt, one, two), bool, Span::ZERO);
        let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
        let join = builder.new_block(&[int]);
        let p = builder.block_params(join)[0];
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        }, acvus_ast::Span::ZERO);
        builder.switch_to(then_block);
        let t = builder.push(constant(10), int, Span::ZERO);
        builder.terminate(jump(join, vec![t]), acvus_ast::Span::ZERO);
        builder.switch_to(else_block);
        let e = builder.push(constant(20), int, Span::ZERO);
        builder.terminate(jump(join, vec![e]), acvus_ast::Span::ZERO);
        builder.switch_to(join);
        builder.terminate(ret(p), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let constants = facts::<ConstantValues>(&f.module, &decls, id);
        assert_eq!(constants.constant(c), Some(&Literal::Bool(true)));
        assert_eq!(*constants.of(e), Constant::Bottom, "the else arm is not executable");
        assert_eq!(constants.constant(p), Some(&Literal::Int(10)));

        let intervals = facts::<IntervalValues>(&f.module, &decls, id);
        assert_eq!(*intervals.of(p), finite(10, 20));
    }

    /// `switch k { 1 => a, 2 => b, _ => d }` on a known `k = 2` takes only
    /// `b`.
    #[test]
    fn a_switch_on_a_known_key_runs_one_arm() {
        let mut f = fixture();
        let decls = declarations(&f.interner, Vec::new(), Types::new());
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let k = builder.push(constant(2), int, Span::ZERO);
        let [a, b, d] = [(); 3].map(|()| builder.new_block(&[]));
        let join = builder.new_block(&[int]);
        let p = builder.block_params(join)[0];
        builder.terminate(Terminator::Switch {
            tag: k,
            arms: vec![(SwitchKey::Int(1), a, Vec::new()), (SwitchKey::Int(2), b, Vec::new())],
            default: Some((d, Vec::new())),
            join,
        }, acvus_ast::Span::ZERO);
        let mut arm_values = Vec::new();
        for (block, n) in [(a, 10), (b, 20), (d, 30)] {
            builder.switch_to(block);
            let value = builder.push(constant(n), int, Span::ZERO);
            arm_values.push(value);
            builder.terminate(jump(join, vec![value]), acvus_ast::Span::ZERO);
        }
        builder.switch_to(join);
        builder.terminate(ret(p), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let constants = facts::<ConstantValues>(&f.module, &decls, id);
        assert_eq!(constants.constant(p), Some(&Literal::Int(20)));
        assert_eq!(*constants.of(arm_values[0]), Constant::Bottom);
        assert_eq!(*constants.of(arm_values[2]), Constant::Bottom);
    }

    /// ```text
    /// entry:            at = 0; hi = 10; s0 = 0; jump header(s0)
    /// header(s):        for counter in at..hi { body } else exit(trip)
    /// body(counter):    one = 1; s1 = s + one; jump header(s1)
    /// exit(trip):       return trip
    /// ```
    /// The counter's interval is what the `For` supplies; the carried `s`
    /// widens at the header, and `s + 1` is clamped to its width.
    #[test]
    fn a_loop_counter_comes_from_the_for_and_a_carried_value_widens() {
        let mut f = fixture();
        let decls = declarations(&f.interner, Vec::new(), Types::new());
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let at = builder.push(constant(0), int, Span::ZERO);
        let hi = builder.push(constant(10), int, Span::ZERO);
        let s0 = builder.push(constant(0), int, Span::ZERO);
        let header = builder.new_block(&[int]);
        let s = builder.block_params(header)[0];
        let loop_body = builder.new_block(&[int]);
        let counter = builder.block_params(loop_body)[0];
        let exit = builder.new_block(&[int]);
        let trip = builder.block_params(exit)[0];
        builder.terminate(jump(header, vec![s0]), acvus_ast::Span::ZERO);
        builder.switch_to(header);
        builder.terminate(Terminator::For {
            source: ForSource::Range { at, hi },
            stages: Stages::lowered(loop_body),
            exit,
            exit_trip: ExitTrip::Defined,
            exit_args: Vec::new(),
        }, acvus_ast::Span::ZERO);
        builder.switch_to(loop_body);
        let one = builder.push(constant(1), int, Span::ZERO);
        let s1 = builder.push(binary(BinOp::Add(Overflow::Trap), s, one), int, Span::ZERO);
        builder.terminate(jump(header, vec![s1]), acvus_ast::Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(trip), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let intervals = facts::<IntervalValues>(&f.module, &decls, id);
        assert_eq!(*intervals.of(counter), finite(0, 9));
        assert_eq!(*intervals.of(trip), finite(10, 10));
        assert_eq!(
            *intervals.of(s),
            Interval::Range { lo: Bound::Finite(0), hi: Bound::PosInfinity },
            "the growing end of the carried value widens to infinity"
        );
        assert_eq!(*intervals.of(s1), finite(1, i128::from(i64::MAX)));

        let constants = facts::<ConstantValues>(&f.module, &decls, id);
        assert_eq!(*constants.of(counter), Constant::Top);
        assert_eq!(constants.constant(trip), Some(&Literal::Int(10)));
        assert_eq!(*constants.of(s), Constant::Top);

        let affine = facts::<AffineValues>(&f.module, &decls, id);
        let remaining = Combination::atom(Symbol::Remaining(counter));
        assert_eq!(
            *affine.of(counter),
            Affine::Exactly(Combination::constant(9).checked_sub(&remaining).unwrap()),
            "the counter is hi - 1 less the steps left"
        );
    }

    /// `f(p0: i64) -> i64` declares `0 ≤ ret` and `ret < p0`; a call
    /// `f(x)` with `x = 7` lies in `[0, 6]`, and is no constant. Its
    /// combination is itself, since no relation makes it equal to one.
    #[test]
    fn a_calls_declared_relations_bound_its_result() {
        let interner = Interner::new();
        let id_f = QualifiedRef::root(interner.intern("f"));

        let mut poly = Types::<Poly>::new();
        let poly_int: TyId<Poly> = poly.intern(TyKind::Int(IntTy::I64));
        let declared = poly.intern(TyKind::Fn {
            params: Box::new([poly_int]),
            ret: poly_int,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(crate::decl::declared_fn_flows()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let relations = vec![
            Relation { left: Term::Const(0), comparison: Comparison::Le, right: Term::Atom(Atom::Result) },
            Relation { left: Term::Atom(Atom::Result), comparison: Comparison::Lt, right: Term::Atom(Atom::Arg(0)) },
        ];
        let scheme = Scheme {
            ty: declared,
            param_names: Box::new([interner.intern("p0")]),
            bounds: Box::new([]),
            impls: Some(Impls {
                concrete: Box::new([]),
                generic: Some(GenericImplSig {
                    facts: DeclaredFacts { relations: relations.into(), ..DeclaredFacts::default() },
                }),
            }),
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            specialized: Box::new([]),
        };
        let decls = declarations(&interner, vec![(id_f, scheme)], poly);

        let mut types = Types::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let callee_ty = types.intern(TyKind::Fn {
            params: Box::new([int]),
            ret: int,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let mut module = Module::new(types);
        let mut builder = module.build_body(false, int);
        let x = builder.push(constant(7), int, Span::ZERO);
        let call = Op::FunctionCall {
            callee: Callee::Extern { id: id_f, r#impl: Some(0), required: Vec::new(), member: crate::decl::PairMember::Uniform },
            callee_ty,
            args: vec![x],
        };
        let r = builder.push(call, int, Span::ZERO);
        builder.terminate(ret(r), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let intervals = facts::<IntervalValues>(&module, &decls, id);
        assert_eq!(*intervals.of(r), finite(0, 6));
        let constants = facts::<ConstantValues>(&module, &decls, id);
        assert_eq!(*constants.of(r), Constant::Top);
        let affine = facts::<AffineValues>(&module, &decls, id);
        assert_eq!(*affine.of(r), Affine::Top);
    }

    /// `succ(p0: i64) -> i64` declares `ret = p0 + 1`; for a parameter
    /// `a`, `r = succ(a)` is `a + 1`, so `d = r - a` is the constant 1,
    /// which no interval of `a` shows.
    #[test]
    fn a_calls_declared_equality_makes_its_result_a_combination_of_its_arguments() {
        let interner = Interner::new();
        let succ = QualifiedRef::root(interner.intern("succ"));

        let mut poly = Types::<Poly>::new();
        let poly_int: TyId<Poly> = poly.intern(TyKind::Int(IntTy::I64));
        let declared = poly.intern(TyKind::Fn {
            params: Box::new([poly_int]),
            ret: poly_int,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(crate::decl::declared_fn_flows()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let plus_one = Term::Add(Box::new(Term::Atom(Atom::Arg(0))), Box::new(Term::Const(1)));
        let relations = vec![Relation { left: Term::Atom(Atom::Result), comparison: Comparison::Eq, right: plus_one }];
        let scheme = Scheme {
            ty: declared,
            param_names: Box::new([interner.intern("p0")]),
            bounds: Box::new([]),
            impls: Some(Impls {
                concrete: Box::new([]),
                generic: Some(GenericImplSig {
                    facts: DeclaredFacts { relations: relations.into(), ..DeclaredFacts::default() },
                }),
            }),
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            specialized: Box::new([]),
        };
        let decls = declarations(&interner, vec![(succ, scheme)], poly);

        let mut types = Types::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let callee_ty = types.intern(TyKind::Fn {
            params: Box::new([int]),
            ret: int,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let mut module = Module::new(types);
        let mut builder = module.build_body(false, int);
        let a = builder.param(interner.intern("a"), int, Span::ZERO);
        let call = Op::FunctionCall {
            callee: Callee::Extern { id: succ, r#impl: Some(0), required: Vec::new(), member: crate::decl::PairMember::Uniform },
            callee_ty,
            args: vec![a],
        };
        let r = builder.push(call, int, Span::ZERO);
        let d = builder.push(binary(BinOp::Sub(Overflow::Trap), r, a), int, Span::ZERO);
        builder.terminate(ret(d), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let affine = facts::<AffineValues>(&module, &decls, id);
        let a_plus_one = Combination::atom(Symbol::Value(a)).checked_add(&Combination::constant(1)).unwrap();
        assert_eq!(*affine.of(r), Affine::Exactly(a_plus_one));
        assert_eq!(*affine.of(d), Affine::Exactly(Combination::constant(1)));
        let intervals = facts::<IntervalValues>(&module, &decls, id);
        assert_eq!(*intervals.of(d), finite(i128::from(i64::MIN), i128::from(i64::MAX)));
    }
}
