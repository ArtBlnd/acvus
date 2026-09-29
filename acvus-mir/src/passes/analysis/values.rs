//! The value analyses: the sparse engine over the constant and the interval
//! domains (passes-contracts: `sccp` reads `ConstantValues`, `bce` reads
//! `IntervalValues`). A value operation is its domain's; every other
//! instruction is `top`, but that a call's result lies where its callee's
//! declared relations put it (extern-fn §3).

use acvus_ast::Literal;

use crate::decl::{Declarations, DeclaredFacts, PostTerm, Postcondition, Relation, Subject};
use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Definition, Inst, Module, Op, Terminator};
use crate::ir::{Callee, ExternImpl, ForSource, SwitchKey, ValueId};
use crate::passes::analysis::{Analysis, AnalysisManager, Executable, SparseAnalysis, SparseFacts, solve_sparse};
use crate::passes::domain::{Constant, ConstantDomain, Interval, IntervalDomain, ValueDomain};
use crate::ty::{CastTy, Concrete, Types};

/// Every value's constant, conditional on the arms a known condition takes
/// (Wegman and Zadeck). Cost: the sparse engine's, with height 3.
pub struct ConstantValues;

/// Every integer value's interval, every arm taken. Cost: the sparse
/// engine's, a parameter of a loop header rising at most three times under
/// widening.
pub struct IntervalValues;

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

impl SparseFacts<ConstantDomain> {
    /// `value`'s one value, where it is known.
    pub fn constant(&self, value: ValueId) -> Option<&Literal> {
        match self.of(value) {
            Constant::Known(literal) => Some(literal),
            Constant::Bottom | Constant::Top => None,
        }
    }
}

impl SparseFacts<IntervalDomain> {
    /// The interval the length of `slice` lies in, as the declared
    /// Relations (extern-fn §3) of the extern impl whose result it is put
    /// it: a call's, or an `AsSlice`'s, whose one argument is its
    /// container. `None` where `slice` is no such result, or where no
    /// relation of its impl bounds the result's length by a term of known
    /// interval. Cost: O(the impl's relations and their terms).
    pub fn declared_len(&self, decls: &Declarations, body: &Body, slice: ValueId) -> Option<Interval> {
        let Definition::Inst(made) = body.definition(slice) else {
            return None;
        };
        let (made_by, args) = match body.inst(made).op() {
            Op::FunctionCall { callee: Callee::Extern { id, r#impl, .. }, args, .. } => {
                (ExternImpl { id: *id, r#impl: *r#impl }, args.as_slice())
            }
            Op::AsSlice { container, r#impl, .. } => (*r#impl, std::slice::from_ref(container)),
            _ => return None,
        };
        let relations = &declared_facts(decls, made_by.id, made_by.r#impl)?.ensures;
        let args: Vec<&Interval> = args.iter().map(|arg| self.of(*arg)).collect();
        let counts = Interval::exactly(0).at_least_lo();
        related(relations, &args, &PostTerm::Len(Subject::Ret))
            .reduce(|len, range| len.meet(&range))
            .map(|len| len.meet(&counts))
    }
}

/// The type a value operation computes at, read from the body's values.
struct OperandTypes<'m> {
    types: &'m Types<Concrete>,
    body: &'m Body,
}

impl OperandTypes<'_> {
    /// `value`'s type, where it is one a value operation computes at.
    fn of(&self, value: ValueId) -> Option<CastTy> {
        CastTy::of_kind(self.types.kind(self.body.ty_of(value)))
    }

    /// The domain's value of a value operation, or `None` where `inst` is
    /// none or its operands have no scalar type (D60).
    fn value_operation<D>(&self, domain: &D, inst: &Inst, operands: &[&D::Value]) -> Option<D::Value>
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
            Terminator::Jump { .. } | Terminator::For { .. } | Terminator::Return { .. } | Terminator::Diverge => {
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
            Op::FunctionCall { callee: Callee::Extern { id, r#impl, .. }, .. } => {
                let facts = declared_facts(self.decls, *id, *r#impl);
                let relations = facts.map_or(&[][..], |facts| &facts.ensures);
                related(relations, operands, &PostTerm::Ret).fold(Interval::full(), |result, range| result.meet(&range))
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

/// The facts the impl a call chose declares: a concrete impl by its index,
/// the generic impl after them. A declaration without impls states none.
///
/// # Panics
/// The call names a function or an impl the declarations do not have.
fn declared_facts(decls: &Declarations, id: acvus_utils::QualifiedRef, r#impl: usize) -> Option<&DeclaredFacts> {
    let scheme = decls.scheme(id).unwrap_or_else(|| panic!("the call names {id:?}, which is not declared"));
    let impls = scheme.impls.as_ref()?;
    Some(match impls.concrete.get(r#impl) {
        Some(concrete) => &concrete.facts,
        None if r#impl == impls.concrete.len() => {
            &impls.generic.as_ref().unwrap_or_else(|| panic!("{id:?} has no impl {impl}", impl = r#impl)).facts
        }
        None => panic!("{id:?} has no impl {impl}", impl = r#impl),
    })
}

/// The range each relation puts `subject` in, where one side is `subject`
/// alone and the other a term of known interval over the call's arguments.
fn related<'r>(
    relations: &'r [Postcondition],
    args: &'r [&Interval],
    subject: &'r PostTerm,
) -> impl Iterator<Item = Interval> + 'r {
    relations.iter().filter_map(move |Postcondition { left, relation, right }| {
        let (term, subject_on_left) = match (left == subject, right == subject) {
            (true, false) => (right, true),
            (false, true) => (left, false),
            (true, true) | (false, false) => return None,
        };
        let term = term_interval(term, args)?;
        let one = Interval::exactly(1);
        Some(match (relation, subject_on_left) {
            (Relation::Eq, _) => term,
            (Relation::Le, true) => term.at_most_hi(),
            (Relation::Lt, true) => term.difference(&one).at_most_hi(),
            (Relation::Le, false) => term.at_least_lo(),
            (Relation::Lt, false) => term.sum(&one).at_least_lo(),
        })
    })
}

/// A term's interval over the call's arguments, where every leaf of it has
/// one: a length is a count, and `old(t)` of an argument is the argument,
/// which a call does not change. A term that reads the result has none.
fn term_interval(term: &PostTerm, args: &[&Interval]) -> Option<Interval> {
    Some(match term {
        PostTerm::Const(n) => Interval::exactly(*n),
        PostTerm::Param(at) => {
            (*args.get(*at).unwrap_or_else(|| panic!("a relation names parameter {at} of {} arguments", args.len())))
                .clone()
        }
        PostTerm::Len(Subject::Param(_)) => Interval::exactly(0).at_least_lo(),
        PostTerm::Ret | PostTerm::Len(Subject::Ret) => return None,
        PostTerm::Add(a, b) => term_interval(a, args)?.sum(&term_interval(b, args)?),
        PostTerm::Sub(a, b) => term_interval(a, args)?.difference(&term_interval(b, args)?),
        PostTerm::Mul(a, b) => term_interval(a, args)?.product(&term_interval(b, args)?),
        PostTerm::Max(a, b) => term_interval(a, args)?.greatest(&term_interval(b, args)?),
        PostTerm::Old(inner) => term_interval(inner, args)?,
    })
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;
    use acvus_utils::{Interner, QualifiedRef};

    use super::*;
    use crate::decl::{Core, GenericImplSig, Impls, LoopMarkers, RawDeclarations, Scheme};
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
            inputs: Vec::new(),
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
    }

    /// `f(p0: i64) -> i64` declares `0 ≤ ret` and `ret < p0`; a call
    /// `f(x)` with `x = 7` lies in `[0, 6]`, and is no constant.
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
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });
        let ensures = vec![
            Postcondition { left: PostTerm::Const(0), relation: Relation::Le, right: PostTerm::Ret },
            Postcondition { left: PostTerm::Ret, relation: Relation::Lt, right: PostTerm::Param(0) },
        ];
        let scheme = Scheme {
            ty: declared,
            param_names: Box::new([interner.intern("p0")]),
            bounds: Box::new([]),
            impls: Some(Impls {
                concrete: Box::new([]),
                generic: Some(GenericImplSig {
                    facts: DeclaredFacts { ensures: ensures.into(), ..DeclaredFacts::default() },
                }),
            }),
            requires: Box::new([]),
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
        });
        let mut module = Module::new(types);
        let mut builder = module.build_body(false, int);
        let x = builder.push(constant(7), int, Span::ZERO);
        let call = Op::FunctionCall {
            callee: Callee::Extern { id: id_f, r#impl: 0, required: Vec::new() },
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
    }
}
