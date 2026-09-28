//! The engines over textbook bodies with known answers, one test domain at a
//! time, and the manager's keeping and invalidation. Bodies are built through
//! the Builder only.

use std::collections::BTreeSet;
use std::rc::Rc;

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, LocalIdOps, QualifiedRef};

use super::*;
use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Changes, Inst, Module, Op, Stages, Terminator};
use crate::ir::{BinOp, Callee, ExitTrip, ForSource, Overflow, UnaryOp, ValueId};
use crate::passes::domain::{JoinSemiLattice, ValueDomain, Widen};
use crate::ty::{
    Calls, CastTy, Contexts, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Reissue, SplitCost, Ty, TyKind, Types,
};

// -- Fixture ------------------------------------------------------------

struct Fixture {
    module: Module,
    interner: Interner,
    int: Ty,
    bool: Ty,
    order: Ty,
    /// A callee type of no parameters returning an `int`, whose effect is
    /// not pure, so a call of it is ordered.
    effectful: Ty,
}

fn fixture() -> Fixture {
    let mut types = Types::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let bool = types.intern(TyKind::Bool);
    let order = types.intern(TyKind::Order);
    let effect = Effect::new(Reissue::Opaque, false, Contexts::new(), Contexts::new(), SplitCost::Unavailable);
    let effectful = types.intern(TyKind::Fn {
        params: Box::new([]),
        ret: int,
        effect: EffectTerm::Known(effect),
        flows: FlowTerm::Known(Flows::none()),
        calls: Calls::Many,
        holds: Holds::Nothing,
    });
    Fixture {
        module: Module::new(types, int, Flows::none()),
        interner: Interner::new(),
        int,
        bool,
        order,
        effectful,
    }
}

fn constant(n: i128) -> Op {
    Op::Const { value: Literal::Int(n) }
}

fn truth(b: bool) -> Op {
    Op::Const { value: Literal::Bool(b) }
}

fn binary(op: BinOp, left: ValueId, right: ValueId) -> Op {
    Op::BinOp { op, left, right }
}

fn add(left: ValueId, right: ValueId) -> Op {
    binary(BinOp::Add(Overflow::Trap), left, right)
}

fn jump(to: BlockId, args: Vec<ValueId>) -> Terminator {
    Terminator::Jump { to, args }
}

fn ret(value: ValueId) -> Terminator {
    Terminator::Return { value, order: None, span: Span::ZERO }
}

fn diamond(cond: ValueId, then_to: BlockId, else_to: BlockId, join: BlockId) -> Terminator {
    Terminator::Diamond { cond, then_to, then_args: Vec::new(), else_to, else_args: Vec::new(), join }
}

fn while_loop(cond: ValueId, body: BlockId, exit: BlockId) -> Terminator {
    Terminator::While { cond, stages: Stages::lowered(body), exit, exit_args: Vec::new() }
}

fn empty_declarations(interner: &Interner) -> Declarations {
    Declarations::verify(RawDeclarations {
        types: Types::new(),
        functions: Vec::new(),
        user_types: Vec::new(),
        casts: Vec::new(),
        conversions: Vec::new(),
        contexts: Vec::new(),
        inputs: Vec::new(),
        loop_markers: LoopMarkers::default(),
        core: Core::of(interner),
    })
    .expect("empty declarations verify")
}

// -- Lattice laws ---------------------------------------------------------

fn joined<L>(a: &L, b: &L) -> L
where
    L: JoinSemiLattice,
{
    let mut out = a.clone();
    out.join(b);
    out
}

/// Join is associative, commutative and idempotent, `bottom` is its unit,
/// `join` reports growth exactly when it changed its receiver, and `le`
/// agrees with join, over every pair and triple of `samples`.
fn assert_lattice_laws<L>(samples: &[L])
where
    L: JoinSemiLattice + std::fmt::Debug,
{
    for a in samples {
        assert_eq!(joined(a, a), *a, "join is idempotent at {a:?}");
        assert_eq!(joined(&L::bottom(), a), *a, "bottom is join's unit at {a:?}");
        assert_eq!(joined(a, &L::bottom()), *a, "bottom is join's unit at {a:?}");
        for b in samples {
            assert_eq!(joined(a, b), joined(b, a), "join commutes at {a:?}, {b:?}");
            assert_eq!(a.le(b), joined(a, b) == *b, "le agrees with join at {a:?}, {b:?}");
            let mut grown = a.clone();
            assert_eq!(grown.join(b), grown != *a, "join reports growth at {a:?}, {b:?}");
            for c in samples {
                assert_eq!(
                    joined(&joined(a, b), c),
                    joined(a, &joined(b, c)),
                    "join associates at {a:?}, {b:?}, {c:?}"
                );
            }
        }
    }
}

// -- A constant domain, finite height --------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Const {
    Bottom,
    Int(i128),
    Bool(bool),
    Top,
}

impl JoinSemiLattice for Const {
    fn bottom() -> Self {
        Const::Bottom
    }

    fn join(&mut self, other: &Self) -> bool {
        let next = match (&*self, other) {
            (_, Const::Bottom) => return false,
            (Const::Bottom, _) => other.clone(),
            (a, b) if a == b => return false,
            _ => Const::Top,
        };
        let grew = *self != next;
        *self = next;
        grew
    }

    fn le(&self, other: &Self) -> bool {
        matches!((self, other), (Const::Bottom, _) | (_, Const::Top)) || self == other
    }
}

/// The constant lattice has height three, so its widening is its join.
impl Widen for Const {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

struct ConstDomain;

impl ValueDomain for ConstDomain {
    type Value = Const;

    fn top(&self) -> Const {
        Const::Top
    }

    fn constant(&self, literal: &Literal) -> Const {
        match literal {
            Literal::Int(n) => Const::Int(*n),
            Literal::Bool(b) => Const::Bool(*b),
            _ => Const::Top,
        }
    }

    fn binary(&self, op: BinOp, left: &Const, right: &Const) -> Const {
        match (op, left, right) {
            (_, Const::Bottom, _) | (_, _, Const::Bottom) => Const::Bottom,
            (BinOp::Add(_), Const::Int(a), Const::Int(b)) => a.checked_add(*b).map_or(Const::Top, Const::Int),
            (BinOp::Lt, Const::Int(a), Const::Int(b)) => Const::Bool(a < b),
            _ => Const::Top,
        }
    }

    fn unary(&self, op: UnaryOp, operand: &Const) -> Const {
        match (op, operand) {
            (_, Const::Bottom) => Const::Bottom,
            (UnaryOp::Not, Const::Bool(b)) => Const::Bool(!b),
            (UnaryOp::Neg(_), Const::Int(n)) => n.checked_neg().map_or(Const::Top, Const::Int),
            _ => Const::Top,
        }
    }

    fn cast(&self, _from: CastTy, _to: CastTy, operand: &Const) -> Const {
        match operand {
            Const::Bottom => Const::Bottom,
            _ => Const::Top,
        }
    }
}

/// Value operations through the domain, anything else `top`; a branch on a
/// known condition runs only its arm, and on `bottom` none yet.
struct Propagate<D> {
    domain: D,
}

/// A value a branch reads: whether it is a known truth.
trait Condition: JoinSemiLattice {
    fn known(&self) -> Option<bool>;
}

/// On `bottom` no arm runs yet, on a known truth only its arm, else both.
fn branch<V>(cond: &V, on_true: BlockId, on_false: BlockId) -> Executable
where
    V: Condition,
{
    if *cond == V::bottom() {
        return Executable::Only(Vec::new());
    }
    match cond.known() {
        Some(true) => Executable::Only(vec![on_true]),
        Some(false) => Executable::Only(vec![on_false]),
        None => Executable::All,
    }
}

impl Condition for Const {
    fn known(&self) -> Option<bool> {
        match self {
            Const::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl<D> SparseAnalysis for Propagate<D>
where
    D: ValueDomain,
    D::Value: Condition,
{
    type Domain = D;

    fn domain(&self) -> &D {
        &self.domain
    }

    fn transfer(&self, inst: &Inst, operands: &[&D::Value]) -> D::Value {
        match (inst.op(), operands) {
            (Op::Const { value }, []) => self.domain.constant(value),
            (Op::BinOp { op, .. }, [left, right]) => self.domain.binary(*op, left, right),
            (Op::UnaryOp { op, .. }, [operand]) => self.domain.unary(*op, operand),
            _ => self.domain.top(),
        }
    }

    fn executable(&self, terminator: &Terminator, operands: &[&D::Value]) -> Executable {
        match terminator {
            Terminator::JumpIf { then_to, else_to, .. } | Terminator::Diamond { then_to, else_to, .. } => {
                branch(operands[0], *then_to, *else_to)
            }
            Terminator::While { stages, exit, .. } => branch(operands[0], stages.body(), *exit),
            _ => Executable::All,
        }
    }

    /// It knows nothing of what a terminator supplies.
    fn supplied(&self, _terminator: &Terminator, _to: BlockId, _index: usize, _operands: &[&D::Value]) -> D::Value {
        self.domain.top()
    }
}

#[test]
fn the_constant_domain_is_a_lattice() {
    assert_lattice_laws(&[Const::Bottom, Const::Int(0), Const::Int(1), Const::Bool(true), Const::Bool(false), Const::Top]);
}

/// `x = 2; y = 3; z = x + y; w = z + a; return w`.
#[test]
fn sparse_folds_a_straight_line() {
    let mut f = fixture();
    let (name, int) = (f.interner.intern("a"), f.int);
    let mut builder = f.module.build_body(false);
    let a = builder.param(name, int);
    let x = builder.push(constant(2), int, Span::ZERO);
    let y = builder.push(constant(3), int, Span::ZERO);
    let z = builder.push(add(x, y), int, Span::ZERO);
    let w = builder.push(add(z, a), int, Span::ZERO);
    builder.terminate(ret(w));
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_sparse(&Propagate { domain: ConstDomain }, f.module.body(id));
    assert_eq!(*facts.of(a), Const::Top, "a parameter comes from outside the body");
    assert_eq!(*facts.of(x), Const::Int(2));
    assert_eq!(*facts.of(y), Const::Int(3));
    assert_eq!(*facts.of(z), Const::Int(5));
    assert_eq!(*facts.of(w), Const::Top);
}

/// `c = true; if c { t = 1 } else { e = 2 }; join(p) = then: t, else: e`:
/// only the then arm runs, so `p` is `t`'s constant and `e` is never
/// defined.
#[test]
fn sparse_runs_only_the_known_arm_of_a_diamond() {
    let mut f = fixture();
    let (int, bool) = (f.int, f.bool);
    let mut builder = f.module.build_body(false);
    let c = builder.push(truth(true), bool, Span::ZERO);
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[int]);
    let p = builder.block_params(join)[0];
    builder.terminate(diamond(c, then_block, else_block, join));
    builder.switch_to(then_block);
    let t = builder.push(constant(1), int, Span::ZERO);
    builder.terminate(jump(join, vec![t]));
    builder.switch_to(else_block);
    let e = builder.push(constant(2), int, Span::ZERO);
    builder.terminate(jump(join, vec![e]));
    builder.switch_to(join);
    builder.terminate(ret(p));
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_sparse(&Propagate { domain: ConstDomain }, f.module.body(id));
    assert_eq!(*facts.of(c), Const::Bool(true));
    assert_eq!(*facts.of(t), Const::Int(1));
    assert_eq!(*facts.of(e), Const::Bottom, "the else arm is not executable");
    assert_eq!(*facts.of(p), Const::Int(1), "the join takes only the executable arm's value");
}

/// The same diamond on an unknown condition joins both arms.
#[test]
fn sparse_joins_both_arms_of_an_unknown_diamond() {
    let mut f = fixture();
    let (name, int, bool) = (f.interner.intern("c"), f.int, f.bool);
    let mut builder = f.module.build_body(false);
    let c = builder.param(name, bool);
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[int]);
    let p = builder.block_params(join)[0];
    builder.terminate(diamond(c, then_block, else_block, join));
    builder.switch_to(then_block);
    let t = builder.push(constant(1), int, Span::ZERO);
    builder.terminate(jump(join, vec![t]));
    builder.switch_to(else_block);
    let e = builder.push(constant(2), int, Span::ZERO);
    builder.terminate(jump(join, vec![e]));
    builder.switch_to(join);
    builder.terminate(ret(p));
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_sparse(&Propagate { domain: ConstDomain }, f.module.body(id));
    assert_eq!(*facts.of(e), Const::Int(2));
    assert_eq!(*facts.of(p), Const::Top);
}

/// A counting loop and its body, built for any test:
///
/// ```text
/// entry:  i0 = 0; k0 = 5; jump header(i0, k0)
/// header(i, k): c = i < n; while c { body } else exit
/// body:   one = 1; i1 = i + one; jump header(i1, k)
/// exit:   return i
/// ```
struct CountingLoop {
    id: BodyId,
    header: BlockId,
    i: ValueId,
    k: ValueId,
    one: ValueId,
    i1: ValueId,
}

fn counting_loop(f: &mut Fixture) -> CountingLoop {
    let (name, int, bool) = (f.interner.intern("n"), f.int, f.bool);
    let mut builder = f.module.build_body(false);
    let n = builder.param(name, int);
    let i0 = builder.push(constant(0), int, Span::ZERO);
    let k0 = builder.push(constant(5), int, Span::ZERO);
    let header = builder.new_block(&[int, int]);
    let [i, k] = builder.block_params(header) else {
        panic!("the header was made with two parameters")
    };
    let (i, k) = (*i, *k);
    let [loop_body, exit] = [(); 2].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header, vec![i0, k0]));
    builder.switch_to(header);
    let c = builder.push(binary(BinOp::Lt, i, n), bool, Span::ZERO);
    builder.terminate(while_loop(c, loop_body, exit));
    builder.switch_to(loop_body);
    let one = builder.push(constant(1), int, Span::ZERO);
    let i1 = builder.push(add(i, one), int, Span::ZERO);
    builder.terminate(jump(header, vec![i1, k]));
    builder.switch_to(exit);
    builder.terminate(ret(i));
    let id = builder.seal().expect("the body has no open state");
    CountingLoop { id, header, i, k, one, i1 }
}

#[test]
fn the_loop_header_is_the_back_edge_target() {
    let mut f = fixture();
    let lp = counting_loop(&mut f);
    let headers = f.module.body(lp.id).loop_headers();
    assert_eq!(headers.into_iter().collect::<Vec<_>>(), vec![lp.header]);
}

/// The carried `i` rises from 0 to `top` at the header; the carried `k`,
/// passed on unchanged, and the loop's own constant stay known.
#[test]
fn sparse_widens_a_loop_carried_value() {
    let mut f = fixture();
    let lp = counting_loop(&mut f);
    let facts = solve_sparse(&Propagate { domain: ConstDomain }, f.module.body(lp.id));
    assert_eq!(*facts.of(lp.i), Const::Top);
    assert_eq!(*facts.of(lp.i1), Const::Top);
    assert_eq!(*facts.of(lp.k), Const::Int(5));
    assert_eq!(*facts.of(lp.one), Const::Int(1));
}

// -- An upper-bound domain, infinite height --------------------------------

/// `AtMost(n)`: the value is at most `n`. Its ascending chains are
/// infinite, so only its widening, which jumps to `Unbounded` on any rise,
/// ends a fixpoint.
#[derive(Debug, Clone, PartialEq)]
enum Bound {
    Bottom,
    AtMost(i128),
    Unbounded,
}

impl JoinSemiLattice for Bound {
    fn bottom() -> Self {
        Bound::Bottom
    }

    fn join(&mut self, other: &Self) -> bool {
        let next = match (&*self, other) {
            (_, Bound::Bottom) => return false,
            (Bound::Bottom, _) | (_, Bound::Unbounded) => other.clone(),
            (Bound::AtMost(a), Bound::AtMost(b)) => Bound::AtMost(*a.max(b)),
            (Bound::Unbounded, _) => Bound::Unbounded,
        };
        let grew = *self != next;
        *self = next;
        grew
    }

    fn le(&self, other: &Self) -> bool {
        match (self, other) {
            (Bound::Bottom, _) | (_, Bound::Unbounded) => true,
            (Bound::AtMost(a), Bound::AtMost(b)) => a <= b,
            (Bound::AtMost(_) | Bound::Unbounded, Bound::Bottom) | (Bound::Unbounded, Bound::AtMost(_)) => false,
        }
    }
}

impl Widen for Bound {
    fn widen(&mut self, next: &Self) -> bool {
        match (&*self, next) {
            (Bound::Bottom, _) => self.join(next),
            _ if next.le(self) => false,
            _ => {
                *self = Bound::Unbounded;
                true
            }
        }
    }
}

impl Condition for Bound {
    fn known(&self) -> Option<bool> {
        None
    }
}

struct BoundDomain;

impl ValueDomain for BoundDomain {
    type Value = Bound;

    fn top(&self) -> Bound {
        Bound::Unbounded
    }

    fn constant(&self, literal: &Literal) -> Bound {
        match literal {
            Literal::Int(n) => Bound::AtMost(*n),
            _ => Bound::Unbounded,
        }
    }

    fn binary(&self, op: BinOp, left: &Bound, right: &Bound) -> Bound {
        match (op, left, right) {
            (_, Bound::Bottom, _) | (_, _, Bound::Bottom) => Bound::Bottom,
            (BinOp::Add(_), Bound::AtMost(a), Bound::AtMost(b)) => {
                a.checked_add(*b).map_or(Bound::Unbounded, Bound::AtMost)
            }
            _ => Bound::Unbounded,
        }
    }

    fn unary(&self, _op: UnaryOp, operand: &Bound) -> Bound {
        match operand {
            Bound::Bottom => Bound::Bottom,
            _ => Bound::Unbounded,
        }
    }

    fn cast(&self, _from: CastTy, _to: CastTy, operand: &Bound) -> Bound {
        match operand {
            Bound::Bottom => Bound::Bottom,
            _ => Bound::Unbounded,
        }
    }
}

#[test]
fn the_bound_domain_is_a_lattice() {
    assert_lattice_laws(&[Bound::Bottom, Bound::AtMost(-1), Bound::AtMost(0), Bound::AtMost(7), Bound::Unbounded]);
}

/// Without widening at the header, `i` would rise through 0, 1, 2, ... and
/// the engine would not end.
#[test]
fn sparse_ends_on_an_infinite_domain_by_widening_at_the_header() {
    let mut f = fixture();
    let lp = counting_loop(&mut f);
    let facts = solve_sparse(&Propagate { domain: BoundDomain }, f.module.body(lp.id));
    assert_eq!(*facts.of(lp.i), Bound::Unbounded);
    assert_eq!(*facts.of(lp.k), Bound::AtMost(5));
    assert_eq!(*facts.of(lp.one), Bound::AtMost(1));
}

// -- What a terminator supplies ---------------------------------------------

/// Constants, and a `For` over `at..hi` whose bounds are known: its counter
/// on the body edge is `at` where the range holds one value, and its trip
/// count on the exit edge is `hi - at` where that is not negative.
struct Ranges;

impl SparseAnalysis for Ranges {
    type Domain = ConstDomain;

    fn domain(&self) -> &ConstDomain {
        &ConstDomain
    }

    fn transfer(&self, inst: &Inst, operands: &[&Const]) -> Const {
        Propagate { domain: ConstDomain }.transfer(inst, operands)
    }

    fn executable(&self, terminator: &Terminator, operands: &[&Const]) -> Executable {
        Propagate { domain: ConstDomain }.executable(terminator, operands)
    }

    fn supplied(&self, terminator: &Terminator, to: BlockId, index: usize, operands: &[&Const]) -> Const {
        let Terminator::For { source: ForSource::Range { .. }, stages, exit, .. } = terminator else {
            return Const::Top;
        };
        let [Const::Int(at), Const::Int(hi), ..] = operands else {
            return Const::Top;
        };
        match (to, index) {
            (to, 0) if to == stages.body() && *hi == at + 1 => Const::Int(*at),
            (to, 0) if to == *exit && hi >= at => Const::Int(hi - at),
            _ => Const::Top,
        }
    }
}

/// ```text
/// entry:  at = 3; hi = 4; jump header
/// header: for counter in at..hi { body } else exit(trip)
/// body(counter): one = 1; x = counter + one; jump header
/// exit(trip): return trip
/// ```
#[test]
fn sparse_takes_a_for_loops_supplied_parameters_from_the_analysis() {
    let mut f = fixture();
    let int = f.int;
    let mut builder = f.module.build_body(false);
    let at = builder.push(constant(3), int, Span::ZERO);
    let hi = builder.push(constant(4), int, Span::ZERO);
    let header = builder.new_block(&[]);
    let loop_body = builder.new_block(&[int]);
    let exit = builder.new_block(&[int]);
    let (counter, trip) = (builder.block_params(loop_body)[0], builder.block_params(exit)[0]);
    builder.terminate(jump(header, Vec::new()));
    builder.switch_to(header);
    builder.terminate(Terminator::For {
        source: ForSource::Range { at, hi },
        stages: Stages::lowered(loop_body),
        exit,
        exit_trip: ExitTrip::Defined,
        exit_args: Vec::new(),
    });
    builder.switch_to(loop_body);
    let one = builder.push(constant(1), int, Span::ZERO);
    let x = builder.push(add(counter, one), int, Span::ZERO);
    builder.terminate(jump(header, Vec::new()));
    builder.switch_to(exit);
    builder.terminate(ret(trip));
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_sparse(&Ranges, f.module.body(id));
    assert_eq!(*facts.of(counter), Const::Int(3));
    assert_eq!(*facts.of(x), Const::Int(4));
    assert_eq!(*facts.of(trip), Const::Int(1));

    let unknowing = solve_sparse(&Propagate { domain: ConstDomain }, f.module.body(id));
    assert_eq!(*unknowing.of(counter), Const::Top, "an analysis that knows nothing states top");
    assert_eq!(*unknowing.of(trip), Const::Top);
}

// -- Order tokens ------------------------------------------------------------

/// `Propagate`, checking that no order token reaches it: a call of no
/// arguments has no operand, a `Merge` is never visited, and a `Return`
/// hands over its value alone.
struct TokenBlind;

impl SparseAnalysis for TokenBlind {
    type Domain = ConstDomain;

    fn domain(&self) -> &ConstDomain {
        &ConstDomain
    }

    fn transfer(&self, inst: &Inst, operands: &[&Const]) -> Const {
        match inst.op() {
            Op::Merge { .. } => panic!("a Merge's result is a token and is not visited"),
            Op::FunctionCall { .. } => assert!(operands.is_empty(), "the consumed token is left out"),
            _ => {}
        }
        Propagate { domain: ConstDomain }.transfer(inst, operands)
    }

    fn executable(&self, terminator: &Terminator, operands: &[&Const]) -> Executable {
        if let Terminator::Return { .. } = terminator {
            assert_eq!(operands.len(), 1, "a Return's order token is left out");
        }
        Propagate { domain: ConstDomain }.executable(terminator, operands)
    }

    fn supplied(&self, _terminator: &Terminator, _to: BlockId, _index: usize, _operands: &[&Const]) -> Const {
        Const::Top
    }
}

/// ```text
/// entry(order): anyorder { r1 = call(); r2 = call() }; k = 7; y = r1 + r2; return y
/// ```
#[test]
fn sparse_analyses_values_and_holds_no_fact_for_a_token() {
    let mut f = fixture();
    let (int, order, effectful, name) = (f.int, f.order, f.effectful, f.interner.intern("f"));
    let call = || Op::FunctionCall { callee: Callee::Direct(QualifiedRef::root(name)), callee_ty: effectful, args: Vec::new() };
    let mut builder = f.module.build_body(true);
    builder.open_anyorder();
    let r1 = builder.push_effect(call(), int, Span::ZERO);
    let r2 = builder.push_effect(call(), int, Span::ZERO);
    builder.close_anyorder();
    let k = builder.push(constant(7), int, Span::ZERO);
    let y = builder.push(add(r1, r2), int, Span::ZERO);
    builder.terminate(ret(y));
    let id = builder.seal().expect("the body has no open state");

    let body = f.module.body(id);
    let merges = body.blocks().flat_map(|(_, block)| block.insts()).filter(|inst| matches!(inst.op(), Op::Merge { .. }));
    assert!(merges.count() > 0, "the anyorder scope gathered its calls with a Merge");
    let facts = solve_sparse(&TokenBlind, body);
    assert_eq!(*facts.of(k), Const::Int(7));
    assert_eq!(*facts.of(r1), Const::Top);
    assert_eq!(*facts.of(y), Const::Top);

    let tokens: Vec<ValueId> = body.values().filter(|value| body.ty_of(*value) == order).collect();
    assert!(tokens.len() >= 4, "the order parameter, two yielded tokens and a merge: {tokens:?}");
    for token in tokens {
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| facts.of(token).clone()));
        assert!(read.is_err(), "{token:?} is a token and has no fact");
    }
}

// -- Dense ------------------------------------------------------------------

/// A set of values; finite over a body, so its widening is its join.
#[derive(Debug, Clone, PartialEq)]
struct Values(BTreeSet<ValueId>);

impl JoinSemiLattice for Values {
    fn bottom() -> Self {
        Values(BTreeSet::new())
    }

    fn join(&mut self, other: &Self) -> bool {
        let before = self.0.len();
        self.0.extend(other.0.iter().copied());
        self.0.len() != before
    }

    fn le(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }
}

impl Widen for Values {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

fn values(of: &[ValueId]) -> Values {
    Values(of.iter().copied().collect())
}

/// Liveness over instructions: a result dies at its definition, and every
/// operand is live before its reader.
struct Liveness;

impl DenseAnalysis for Liveness {
    type State = Values;

    const DIRECTION: Direction = Direction::Backward;

    fn entry(&self, _body: &Body) -> Values {
        Values::bottom()
    }

    fn transfer(&self, inst: &Inst, state: &mut Values) {
        if let Some(result) = inst.result() {
            state.0.remove(&result);
        }
        state.0.extend(inst.operands());
    }

    fn transfer_terminator(&self, terminator: &Terminator, state: &mut Values) {
        state.0.extend(terminator.operands());
    }
}

#[test]
fn the_value_set_is_a_lattice() {
    let [a, b, c] = [0, 1, 2].map(ValueId::from_raw);
    assert_lattice_laws(&[values(&[]), values(&[a]), values(&[b]), values(&[a, b]), values(&[a, b, c])]);
}

/// ```text
/// entry(a, b, c): x = a + a; jump header
/// header: while c { body } else exit
/// body:   y = x + b; jump header
/// exit:   return a
/// ```
/// `x` and `b` are live at the end of the body only through the back edge.
#[test]
fn dense_liveness_runs_backward_around_a_loop() {
    let mut f = fixture();
    let (na, nb, nc, int, bool) =
        (f.interner.intern("a"), f.interner.intern("b"), f.interner.intern("c"), f.int, f.bool);
    let mut builder = f.module.build_body(false);
    let a = builder.param(na, int);
    let b = builder.param(nb, int);
    let c = builder.param(nc, bool);
    let entry = builder.current_block();
    let x = builder.push(add(a, a), int, Span::ZERO);
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header, Vec::new()));
    builder.switch_to(header);
    builder.terminate(while_loop(c, loop_body, exit));
    builder.switch_to(loop_body);
    builder.push(add(x, b), int, Span::ZERO);
    builder.terminate(jump(header, Vec::new()));
    builder.switch_to(exit);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_dense(&Liveness, f.module.body(id));
    let around = values(&[a, b, c, x]);
    assert_eq!(*facts.at(exit), values(&[]));
    assert_eq!(*facts.at(header), around);
    assert_eq!(*facts.at(loop_body), around);
    assert_eq!(*facts.at(entry), around);
}

/// The values some path to a block's start has defined: a forward union.
struct Defined;

impl DenseAnalysis for Defined {
    type State = Values;

    const DIRECTION: Direction = Direction::Forward;

    fn entry(&self, body: &Body) -> Values {
        Values(body.params().iter().map(|(_, value)| *value).collect())
    }

    fn transfer(&self, inst: &Inst, state: &mut Values) {
        state.0.extend(inst.result());
    }

    fn transfer_terminator(&self, _terminator: &Terminator, _state: &mut Values) {}
}

#[test]
fn dense_runs_forward_over_a_diamond() {
    let mut f = fixture();
    let (nc, int) = (f.interner.intern("c"), f.int);
    let bool = f.bool;
    let mut builder = f.module.build_body(false);
    let c = builder.param(nc, bool);
    let x = builder.push(constant(0), int, Span::ZERO);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join));
    builder.switch_to(then_block);
    let t = builder.push(constant(1), int, Span::ZERO);
    builder.terminate(jump(join, Vec::new()));
    builder.switch_to(else_block);
    let e = builder.push(constant(2), int, Span::ZERO);
    builder.terminate(jump(join, Vec::new()));
    builder.switch_to(join);
    builder.terminate(ret(x));
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_dense(&Defined, f.module.body(id));
    assert_eq!(*facts.at(f.module.body(id).entry()), values(&[c]));
    assert_eq!(*facts.at(then_block), values(&[c, x]));
    assert_eq!(*facts.at(else_block), values(&[c, x]));
    assert_eq!(*facts.at(join), values(&[c, x, t, e]), "the join takes both arms");
}

// -- Manager ----------------------------------------------------------------

/// Bodies of one block that return their parameter.
fn identity_bodies<const N: usize>(f: &mut Fixture) -> [BodyId; N] {
    let (name, int) = (f.interner.intern("a"), f.int);
    [(); N].map(|()| {
        let mut builder = f.module.build_body(false);
        let a = builder.param(name, int);
        builder.terminate(ret(a));
        builder.seal().expect("the body has no open state")
    })
}

/// How many blocks the body has.
struct BlockCount;

impl Analysis for BlockCount {
    type Facts = usize;

    const READS: &'static [Aspect] = &[Aspect::CfgShape];

    fn compute(_am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> usize {
        module.body(body).blocks().count()
    }
}

/// The block count, read through the manager, and the instruction count.
struct Size;

struct SizeFacts {
    blocks: Rc<usize>,
    insts: usize,
}

impl Analysis for Size {
    type Facts = SizeFacts;

    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> SizeFacts {
        let blocks = am.get::<BlockCount>(module, body);
        let insts = module.body(body).blocks().map(|(_, block)| block.insts().len()).sum();
        SizeFacts { blocks, insts }
    }
}

/// Requests itself through `Echo`.
struct Narcissus;

impl Analysis for Narcissus {
    type Facts = ();

    const READS: &'static [Aspect] = &[];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) {
        am.get::<Echo>(module, body);
    }
}

struct Echo;

impl Analysis for Echo {
    type Facts = ();

    const READS: &'static [Aspect] = &[];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) {
        am.get::<Narcissus>(module, body);
    }
}

#[test]
#[should_panic(expected = "Narcissus for BodyId(0) was requested while it is being computed")]
fn an_analysis_that_depends_on_itself_is_refused() {
    let mut f = fixture();
    let [body] = identity_bodies(&mut f);
    let decls = empty_declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    am.get::<Narcissus>(&f.module, body);
}

fn changed(aspect: Aspect) -> Changes {
    let mut changes = Changes::default();
    changes.record(aspect);
    changes
}

#[test]
fn a_kept_analysis_is_computed_once() {
    let mut f = fixture();
    let [body] = identity_bodies(&mut f);
    let decls = empty_declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    let first = am.get::<BlockCount>(&f.module, body);
    let second = am.get::<BlockCount>(&f.module, body);
    assert_eq!(*first, 1);
    assert!(Rc::ptr_eq(&first, &second), "the second request returns the kept facts");
}

#[test]
fn an_analysis_requests_another_through_the_manager() {
    let mut f = fixture();
    let [body] = identity_bodies(&mut f);
    let decls = empty_declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    let size = am.get::<Size>(&f.module, body);
    assert_eq!(*size.blocks, 1);
    assert_eq!(size.insts, 0);
    let blocks = am.get::<BlockCount>(&f.module, body);
    assert!(Rc::ptr_eq(&size.blocks, &blocks), "the requested analysis is kept by the manager");
}

#[test]
fn invalidation_drops_exactly_the_analyses_whose_reads_changed() {
    let mut f = fixture();
    let [body, other] = identity_bodies(&mut f);
    let decls = empty_declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    let blocks = am.get::<BlockCount>(&f.module, body);
    let size = am.get::<Size>(&f.module, body);
    let other_blocks = am.get::<BlockCount>(&f.module, other);

    am.invalidate(body, &changed(Aspect::Types));
    assert!(Rc::ptr_eq(&blocks, &am.get::<BlockCount>(&f.module, body)), "a disjoint aspect keeps it");
    assert!(Rc::ptr_eq(&size, &am.get::<Size>(&f.module, body)), "a disjoint aspect keeps it");

    am.invalidate(body, &changed(Aspect::Instructions));
    assert!(Rc::ptr_eq(&blocks, &am.get::<BlockCount>(&f.module, body)), "BlockCount reads no instruction");
    assert!(!Rc::ptr_eq(&size, &am.get::<Size>(&f.module, body)), "an intersecting aspect drops it");

    am.invalidate(body, &changed(Aspect::CfgShape));
    assert!(!Rc::ptr_eq(&blocks, &am.get::<BlockCount>(&f.module, body)), "an intersecting aspect drops it");
    assert!(
        Rc::ptr_eq(&other_blocks, &am.get::<BlockCount>(&f.module, other)),
        "another body's analyses are kept"
    );
}
