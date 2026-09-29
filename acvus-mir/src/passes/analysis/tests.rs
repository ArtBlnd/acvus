//! The engines over textbook bodies with known answers, one test domain at a
//! time, and the manager's keeping and invalidation. Bodies are built through
//! the Builder only.

use std::collections::BTreeSet;
use std::rc::Rc;

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, LocalIdOps, QualifiedRef};

use super::*;
use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Changes, Edge, Inst, Module, Op, Stages, Terminator};
use crate::ir::cfg::Site;
use crate::ir::{BinOp, Callee, ExitTrip, ForSource, Overflow, PathSeg, RefTarget, UnaryOp, ValueId};
use crate::passes::domain::{JoinSemiLattice, ValueDomain, Widen};
use crate::ty::{
    Alignment, Calls, CastTy, Contexts, Effect, EffectTerm, Flow, FlowEnd, FlowTerm, Flows, Holds, IntTy, Mutability,
    Reissue, SplitCost, Ty, TyKind, TypeArg, Types,
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
        values: (),
    });
    Fixture {
        module: Module::new(types),
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
        lang_impls: Vec::new(),
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

    fn constant(&self, literal: &Literal, _ty: CastTy) -> Const {
        match literal {
            Literal::Int(n) => Const::Int(*n),
            Literal::Bool(b) => Const::Bool(*b),
            _ => Const::Top,
        }
    }

    fn binary(&self, op: BinOp, _ty: CastTy, left: &Const, right: &Const) -> Const {
        match (op, left, right) {
            (_, Const::Bottom, _) | (_, _, Const::Bottom) => Const::Bottom,
            (BinOp::Add(_), Const::Int(a), Const::Int(b)) => a.checked_add(*b).map_or(Const::Top, Const::Int),
            (BinOp::Lt, Const::Int(a), Const::Int(b)) => Const::Bool(a < b),
            _ => Const::Top,
        }
    }

    fn unary(&self, op: UnaryOp, _ty: CastTy, operand: &Const) -> Const {
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
struct Propagate<'m, D> {
    domain: D,
    scalars: Scalars<'m>,
}

/// The scalar type of each value of one body, which a value operation is
/// handed.
#[derive(Clone, Copy)]
struct Scalars<'m> {
    module: &'m Module,
    body: &'m Body,
}

impl<'m> Scalars<'m> {
    fn of(module: &'m Module, id: BodyId) -> Scalars<'m> {
        Scalars { module, body: module.body(id) }
    }

    fn ty(&self, value: ValueId) -> CastTy {
        CastTy::of_kind(self.module.types().kind(self.body.ty_of(value)))
            .unwrap_or_else(|| panic!("{value:?} is a scalar in these bodies"))
    }
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

impl<D> SparseAnalysis for Propagate<'_, D>
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
            (Op::Const { value }, []) => {
                let result = inst.result().expect("a constant yields a value");
                self.domain.constant(value, self.scalars.ty(result))
            }
            (Op::BinOp { op, left: of, .. }, [left, right]) => {
                self.domain.binary(*op, self.scalars.ty(*of), left, right)
            }
            (Op::UnaryOp { op, operand: of }, [operand]) => self.domain.unary(*op, self.scalars.ty(*of), operand),
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
    let mut builder = f.module.build_body(false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    let x = builder.push(constant(2), int, Span::ZERO);
    let y = builder.push(constant(3), int, Span::ZERO);
    let z = builder.push(add(x, y), int, Span::ZERO);
    let w = builder.push(add(z, a), int, Span::ZERO);
    builder.terminate(ret(w), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_sparse(&Propagate { domain: ConstDomain, scalars: Scalars::of(&f.module, id) }, f.module.body(id));
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
    let mut builder = f.module.build_body(false, f.int);
    let c = builder.push(truth(true), bool, Span::ZERO);
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[int]);
    let p = builder.block_params(join)[0];
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    let t = builder.push(constant(1), int, Span::ZERO);
    builder.terminate(jump(join, vec![t]), acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    let e = builder.push(constant(2), int, Span::ZERO);
    builder.terminate(jump(join, vec![e]), acvus_ast::Span::ZERO);
    builder.switch_to(join);
    builder.terminate(ret(p), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_sparse(&Propagate { domain: ConstDomain, scalars: Scalars::of(&f.module, id) }, f.module.body(id));
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
    let mut builder = f.module.build_body(false, f.int);
    let c = builder.param(name, bool, acvus_ast::Span::ZERO);
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[int]);
    let p = builder.block_params(join)[0];
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    let t = builder.push(constant(1), int, Span::ZERO);
    builder.terminate(jump(join, vec![t]), acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    let e = builder.push(constant(2), int, Span::ZERO);
    builder.terminate(jump(join, vec![e]), acvus_ast::Span::ZERO);
    builder.switch_to(join);
    builder.terminate(ret(p), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_sparse(&Propagate { domain: ConstDomain, scalars: Scalars::of(&f.module, id) }, f.module.body(id));
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
    let mut builder = f.module.build_body(false, f.int);
    let n = builder.param(name, int, acvus_ast::Span::ZERO);
    let i0 = builder.push(constant(0), int, Span::ZERO);
    let k0 = builder.push(constant(5), int, Span::ZERO);
    let header = builder.new_block(&[int, int]);
    let [i, k] = builder.block_params(header) else {
        panic!("the header was made with two parameters")
    };
    let (i, k) = (*i, *k);
    let [loop_body, exit] = [(); 2].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header, vec![i0, k0]), acvus_ast::Span::ZERO);
    builder.switch_to(header);
    let c = builder.push(binary(BinOp::Lt, i, n), bool, Span::ZERO);
    builder.terminate(while_loop(c, loop_body, exit), acvus_ast::Span::ZERO);
    builder.switch_to(loop_body);
    let one = builder.push(constant(1), int, Span::ZERO);
    let i1 = builder.push(add(i, one), int, Span::ZERO);
    builder.terminate(jump(header, vec![i1, k]), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    builder.terminate(ret(i), acvus_ast::Span::ZERO);
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
    let facts = solve_sparse(&Propagate { domain: ConstDomain, scalars: Scalars::of(&f.module, lp.id) }, f.module.body(lp.id));
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

    fn constant(&self, literal: &Literal, _ty: CastTy) -> Bound {
        match literal {
            Literal::Int(n) => Bound::AtMost(*n),
            _ => Bound::Unbounded,
        }
    }

    fn binary(&self, op: BinOp, _ty: CastTy, left: &Bound, right: &Bound) -> Bound {
        match (op, left, right) {
            (_, Bound::Bottom, _) | (_, _, Bound::Bottom) => Bound::Bottom,
            (BinOp::Add(_), Bound::AtMost(a), Bound::AtMost(b)) => {
                a.checked_add(*b).map_or(Bound::Unbounded, Bound::AtMost)
            }
            _ => Bound::Unbounded,
        }
    }

    fn unary(&self, _op: UnaryOp, _ty: CastTy, operand: &Bound) -> Bound {
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
    let facts = solve_sparse(&Propagate { domain: BoundDomain, scalars: Scalars::of(&f.module, lp.id) }, f.module.body(lp.id));
    assert_eq!(*facts.of(lp.i), Bound::Unbounded);
    assert_eq!(*facts.of(lp.k), Bound::AtMost(5));
    assert_eq!(*facts.of(lp.one), Bound::AtMost(1));
}

// -- What a terminator supplies ---------------------------------------------

/// Constants, and a `For` over `at..hi` whose bounds are known: its counter
/// on the body edge is `at` where the range holds one value, and its trip
/// count on the exit edge is `hi - at` where that is not negative.
struct Ranges<'m>(Scalars<'m>);

impl SparseAnalysis for Ranges<'_> {
    type Domain = ConstDomain;

    fn domain(&self) -> &ConstDomain {
        &ConstDomain
    }

    fn transfer(&self, inst: &Inst, operands: &[&Const]) -> Const {
        Propagate { domain: ConstDomain, scalars: self.0 }.transfer(inst, operands)
    }

    fn executable(&self, terminator: &Terminator, operands: &[&Const]) -> Executable {
        Propagate { domain: ConstDomain, scalars: self.0 }.executable(terminator, operands)
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
    let mut builder = f.module.build_body(false, f.int);
    let at = builder.push(constant(3), int, Span::ZERO);
    let hi = builder.push(constant(4), int, Span::ZERO);
    let header = builder.new_block(&[]);
    let loop_body = builder.new_block(&[int]);
    let exit = builder.new_block(&[int]);
    let (counter, trip) = (builder.block_params(loop_body)[0], builder.block_params(exit)[0]);
    builder.terminate(jump(header, Vec::new()), acvus_ast::Span::ZERO);
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
    let x = builder.push(add(counter, one), int, Span::ZERO);
    builder.terminate(jump(header, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    builder.terminate(ret(trip), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let facts = solve_sparse(&Ranges(Scalars::of(&f.module, id)), f.module.body(id));
    assert_eq!(*facts.of(counter), Const::Int(3));
    assert_eq!(*facts.of(x), Const::Int(4));
    assert_eq!(*facts.of(trip), Const::Int(1));

    let unknowing = solve_sparse(&Propagate { domain: ConstDomain, scalars: Scalars::of(&f.module, id) }, f.module.body(id));
    assert_eq!(*unknowing.of(counter), Const::Top, "an analysis that knows nothing states top");
    assert_eq!(*unknowing.of(trip), Const::Top);
}

// -- Order tokens ------------------------------------------------------------

/// `Propagate`, checking that no order token reaches it: a call of no
/// arguments has no operand, a `Merge` is never visited, and a `Return`
/// hands over its value alone.
struct TokenBlind<'m>(Scalars<'m>);

impl SparseAnalysis for TokenBlind<'_> {
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
        Propagate { domain: ConstDomain, scalars: self.0 }.transfer(inst, operands)
    }

    fn executable(&self, terminator: &Terminator, operands: &[&Const]) -> Executable {
        if let Terminator::Return { .. } = terminator {
            assert_eq!(operands.len(), 1, "a Return's order token is left out");
        }
        Propagate { domain: ConstDomain, scalars: self.0 }.executable(terminator, operands)
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
    let mut builder = f.module.build_body(true, f.int);
    builder.open_anyorder();
    let r1 = builder.push_effect(call(), int, Span::ZERO);
    let r2 = builder.push_effect(call(), int, Span::ZERO);
    builder.close_anyorder();
    let k = builder.push(constant(7), int, Span::ZERO);
    let y = builder.push(add(r1, r2), int, Span::ZERO);
    builder.terminate(ret(y), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.module.body(id);
    let merges = body.blocks().flat_map(|(_, block)| block.insts()).filter(|inst| matches!(inst.op(), Op::Merge { .. }));
    assert!(merges.count() > 0, "the anyorder scope gathered its calls with a Merge");
    let facts = solve_sparse(&TokenBlind(Scalars::of(&f.module, id)), body);
    assert_eq!(*facts.of(k), Const::Int(7));
    assert_eq!(*facts.of(r1), Const::Top);
    assert_eq!(*facts.of(y), Const::Top);

    let tokens: Vec<ValueId> = body.values().filter(|value| body.is_token(*value)).collect();
    assert!(
        body.values().all(|value| body.is_token(value) == (body.ty_of(value) == order)),
        "exactly the values of Order type are tokens"
    );
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

    fn transfer_edge(&self, edge: &Edge<'_>, target_params: &[ValueId], state: &mut Values) {
        for param in target_params {
            state.0.remove(param);
        }
        state.0.extend(edge.args.iter().copied());
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
    let mut builder = f.module.build_body(false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let b = builder.param(nb, int, acvus_ast::Span::ZERO);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let x = builder.push(add(a, a), int, Span::ZERO);
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(header);
    builder.terminate(while_loop(c, loop_body, exit), acvus_ast::Span::ZERO);
    builder.switch_to(loop_body);
    builder.push(add(x, b), int, Span::ZERO);
    builder.terminate(jump(header, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    builder.terminate(ret(a), acvus_ast::Span::ZERO);
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

    fn transfer_edge(&self, _edge: &Edge<'_>, target_params: &[ValueId], state: &mut Values) {
        state.0.extend(target_params.iter().copied());
    }
}

#[test]
fn dense_runs_forward_over_a_diamond() {
    let mut f = fixture();
    let (nc, int) = (f.interner.intern("c"), f.int);
    let bool = f.bool;
    let mut builder = f.module.build_body(false, f.int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let x = builder.push(constant(0), int, Span::ZERO);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    let t = builder.push(constant(1), int, Span::ZERO);
    builder.terminate(jump(join, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    let e = builder.push(constant(2), int, Span::ZERO);
    builder.terminate(jump(join, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(join);
    builder.terminate(ret(x), acvus_ast::Span::ZERO);
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
        let mut builder = f.module.build_body(false, f.int);
        let a = builder.param(name, int, acvus_ast::Span::ZERO);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
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

// -- Dominators, post-dominators, loops --------------------------------------

/// The facts of `A` for `body`, through a manager over no declarations.
/// The site of the instruction at place `index` of `block`.
fn site_at(module: &Module, body: BodyId, block: BlockId, index: usize) -> Site {
    Site::Inst(module.body(body).inst_at(block, index).expect("the block holds that instruction"))
}

/// The point before the instruction at place `index` of `block`, or its
/// end where `index` is its length.
fn point_at(module: &Module, body: BodyId, block: BlockId, index: usize) -> Point {
    let body = module.body(body);
    match body.inst_at(block, index) {
        Some(inst) => Point::Before(inst),
        None => {
            assert_eq!(index, body.inst_count(block), "{block:?} has no place {index}");
            Point::End(block)
        }
    }
}

fn facts_of<A>(f: &Fixture, body: BodyId) -> Rc<A::Facts>
where
    A: Analysis,
{
    let decls = empty_declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    am.get::<A>(&f.module, body)
}

struct DiamondBody {
    id: BodyId,
    entry: BlockId,
    then_block: BlockId,
    else_block: BlockId,
    join: BlockId,
}

/// ```text
/// entry(c): x = 0; diamond c { then } else { else } join
/// then:     t = 1; jump join
/// else:     jump join
/// join:     return x
/// ```
fn diamond_body(f: &mut Fixture) -> DiamondBody {
    let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
    let mut builder = f.module.build_body(false, f.int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let x = builder.push(constant(0), int, Span::ZERO);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    builder.push(constant(1), int, Span::ZERO);
    builder.terminate(jump(join, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    builder.terminate(jump(join, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(join);
    builder.terminate(ret(x), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    DiamondBody { id, entry, then_block, else_block, join }
}

#[test]
fn dominators_of_a_diamond() {
    let mut f = fixture();
    let d = diamond_body(&mut f);
    let dom = facts_of::<Dominators>(&f, d.id);
    assert_eq!(dom.idom(d.entry), None);
    for block in [d.then_block, d.else_block, d.join] {
        assert_eq!(dom.idom(block), Some(d.entry), "{block:?} is entered only through the entry");
        assert!(dom.dominates(d.entry, block));
        assert!(!dom.dominates(block, d.entry));
    }
    assert!(!dom.dominates(d.then_block, d.join), "the join is reached through the else arm too");
    assert!(dom.dominates(d.join, d.join), "a block dominates itself");

    assert!(dom.point_dominates(f.module.body(d.id), point_at(&f.module, d.id, d.entry, 0), Point::End(d.entry)));
    assert!(!dom.point_dominates(f.module.body(d.id), Point::End(d.entry), point_at(&f.module, d.id, d.entry, 0)));
    assert!(dom.point_dominates(f.module.body(d.id), point_at(&f.module, d.id, d.entry, 1), point_at(&f.module, d.id, d.entry, 1)), "a point dominates itself");
    assert!(dom.point_dominates(f.module.body(d.id), Point::End(d.entry), point_at(&f.module, d.id, d.join, 0)));
    assert!(!dom.point_dominates(f.module.body(d.id), Point::End(d.then_block), point_at(&f.module, d.id, d.join, 0)));
}

#[test]
fn post_dominators_of_a_diamond() {
    let mut f = fixture();
    let d = diamond_body(&mut f);
    let post = facts_of::<PostDominators>(&f, d.id);
    for block in [d.entry, d.then_block, d.else_block] {
        assert_eq!(post.ipdom(block), Some(d.join), "every path from {block:?} passes the join");
        assert!(post.post_dominates(d.join, block));
    }
    assert_eq!(post.ipdom(d.join), None, "the join returns: its immediate post-dominator is the exit");
    assert!(post.reaches_exit(d.join));
    assert!(!post.post_dominates(d.then_block, d.entry), "the else arm avoids the then arm");
    assert!(!post.post_dominates(d.entry, d.join));

    assert!(post.point_post_dominates(f.module.body(d.id), Point::End(d.entry), point_at(&f.module, d.id, d.entry, 0)));
    assert!(!post.point_post_dominates(f.module.body(d.id), point_at(&f.module, d.id, d.entry, 0), Point::End(d.entry)));
    assert!(post.point_post_dominates(f.module.body(d.id), point_at(&f.module, d.id, d.join, 0), Point::End(d.entry)));
    assert!(!post.point_post_dominates(f.module.body(d.id), point_at(&f.module, d.id, d.then_block, 0), Point::End(d.entry)));
}

struct WhileBody {
    id: BodyId,
    entry: BlockId,
    header: BlockId,
    loop_body: BlockId,
    exit: BlockId,
}

/// ```text
/// entry(c): jump header
/// header:   while c { body } else exit
/// body:     jump header
/// exit:     return 0
/// ```
fn while_body(f: &mut Fixture) -> WhileBody {
    let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
    let mut builder = f.module.build_body(false, int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(header);
    builder.terminate(while_loop(c, loop_body, exit), acvus_ast::Span::ZERO);
    builder.switch_to(loop_body);
    builder.terminate(jump(header, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    WhileBody { id, entry, header, loop_body, exit }
}

#[test]
fn dominators_and_post_dominators_of_a_loop() {
    let mut f = fixture();
    let w = while_body(&mut f);
    let dom = facts_of::<Dominators>(&f, w.id);
    assert_eq!(dom.idom(w.header), Some(w.entry));
    assert_eq!(dom.idom(w.loop_body), Some(w.header));
    assert_eq!(dom.idom(w.exit), Some(w.header));
    assert!(!dom.dominates(w.loop_body, w.header), "the back edge's source does not dominate its target");

    let post = facts_of::<PostDominators>(&f, w.id);
    assert_eq!(post.ipdom(w.entry), Some(w.header));
    assert_eq!(post.ipdom(w.loop_body), Some(w.header), "every way out of the body returns to the header");
    assert_eq!(post.ipdom(w.header), Some(w.exit));
    assert!(!post.post_dominates(w.loop_body, w.header), "the header may leave without running the body");
}

/// ```text
/// entry(c): diamond c { then } else { else } join
/// then:     diverge
/// else:     jump join
/// join:     return 0
/// ```
#[test]
fn a_diverging_arm_reaches_the_exit_without_the_join() {
    let mut f = fixture();
    let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
    let mut builder = f.module.build_body(false, int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    builder.terminate(Terminator::Diverge, acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    builder.terminate(jump(join, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(join);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let dom = facts_of::<Dominators>(&f, id);
    assert_eq!(dom.idom(then_block), Some(entry));
    assert_eq!(dom.idom(join), Some(else_block), "the join's one way in is the else arm");

    let post = facts_of::<PostDominators>(&f, id);
    assert!(post.reaches_exit(then_block));
    assert_eq!(post.ipdom(then_block), None, "a diverging block's immediate post-dominator is the exit");
    assert_eq!(post.ipdom(entry), None, "the two arms meet only at the exit");
    assert!(!post.post_dominates(join, entry), "the diverging arm leaves without the join");
    assert_eq!(post.ipdom(else_block), Some(join));
}

/// ```text
/// entry: jump spin
/// spin:  jump spin
/// ```
#[test]
fn a_block_that_reaches_no_exit_is_post_dominated_by_itself_alone() {
    let mut f = fixture();
    let mut builder = f.module.build_body(false, f.int);
    let entry = builder.current_block();
    let spin = builder.new_block(&[]);
    builder.terminate(jump(spin, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(spin);
    builder.terminate(jump(spin, Vec::new()), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let post = facts_of::<PostDominators>(&f, id);
    assert!(!post.reaches_exit(entry) && !post.reaches_exit(spin));
    assert_eq!(post.ipdom(entry), None);
    assert!(!post.post_dominates(spin, entry), "not vacuously: the runs from the entry never end");
    assert!(post.post_dominates(spin, spin));
}

struct NestedLoops {
    id: BodyId,
    entry: BlockId,
    outer: BlockId,
    outer_body: BlockId,
    inner: BlockId,
    inner_body: BlockId,
    outer_latch: BlockId,
    exit: BlockId,
}

/// ```text
/// entry(c):    jump outer
/// outer:       while c { outer_body } else exit
/// outer_body:  jump inner
/// inner:       while c { inner_body } else outer_latch
/// inner_body:  jump inner
/// outer_latch: jump outer
/// exit:        return 0
/// ```
fn nested_loops(f: &mut Fixture) -> NestedLoops {
    let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
    let mut builder = f.module.build_body(false, int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let [outer, outer_body, inner, inner_body, outer_latch, exit] = [(); 6].map(|()| builder.new_block(&[]));
    builder.terminate(jump(outer, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(outer);
    builder.terminate(while_loop(c, outer_body, exit), acvus_ast::Span::ZERO);
    builder.switch_to(outer_body);
    builder.terminate(jump(inner, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(inner);
    builder.terminate(while_loop(c, inner_body, outer_latch), acvus_ast::Span::ZERO);
    builder.switch_to(inner_body);
    builder.terminate(jump(inner, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(outer_latch);
    builder.terminate(jump(outer, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    NestedLoops { id, entry, outer, outer_body, inner, inner_body, outer_latch, exit }
}

#[test]
fn the_loop_tree_nests_an_inner_loop_in_an_outer_one_with_their_preheaders() {
    let mut f = fixture();
    let n = nested_loops(&mut f);
    let tree = facts_of::<LoopTree>(&f, n.id);
    assert_eq!(tree.loops().count(), 2);
    let outer = tree.innermost(n.outer).expect("the outer header is in a loop");
    let inner = tree.innermost(n.inner_body).expect("the inner body is in a loop");
    assert_ne!(outer, inner);

    assert_eq!(tree.header(outer), n.outer);
    assert_eq!(tree.header(inner), n.inner);
    let mut outer_blocks = tree.blocks(outer).to_vec();
    outer_blocks.sort();
    let mut expected = vec![n.outer, n.outer_body, n.inner, n.inner_body, n.outer_latch];
    expected.sort();
    assert_eq!(outer_blocks, expected);
    assert_eq!(tree.blocks(inner), [n.inner, n.inner_body]);
    assert!(!tree.contains(outer, n.exit) && !tree.contains(outer, n.entry));
    assert!(tree.contains(outer, n.inner_body) && !tree.contains(inner, n.outer_body));

    assert_eq!(tree.parent(inner), Some(outer));
    assert_eq!(tree.parent(outer), None);
    assert_eq!(tree.innermost(n.outer_body), Some(outer));
    assert_eq!(tree.innermost(n.inner), Some(inner));
    assert_eq!(tree.innermost(n.entry), None);
    assert_eq!(tree.innermost(n.exit), None);

    assert_eq!(tree.preheader(outer), Some(n.entry));
    assert_eq!(tree.preheader(inner), Some(n.outer_body));
}

/// ```text
/// entry(c): diamond c { left } else { right } header
/// left:     jump header
/// right:    jump header
/// header:   while c { body } else exit
/// body:     jump header
/// exit:     return 0
/// ```
#[test]
fn a_header_entered_from_two_blocks_has_no_preheader() {
    let mut f = fixture();
    let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
    let mut builder = f.module.build_body(false, int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let [left, right, header, loop_body, exit] = [(); 5].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, left, right, header), acvus_ast::Span::ZERO);
    for arm in [left, right] {
        builder.switch_to(arm);
        builder.terminate(jump(header, Vec::new()), acvus_ast::Span::ZERO);
    }
    builder.switch_to(header);
    builder.terminate(while_loop(c, loop_body, exit), acvus_ast::Span::ZERO);
    builder.switch_to(loop_body);
    builder.terminate(jump(header, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let tree = facts_of::<LoopTree>(&f, id);
    let [the_loop] = tree.loops().collect::<Vec<_>>()[..] else {
        panic!("the body has one loop")
    };
    assert_eq!(tree.header(the_loop), header);
    assert_eq!(tree.preheader(the_loop), None);
}

// -- Liveness ------------------------------------------------------------------

/// ```text
/// entry(n):  i0 = 0; jump header(i0)
/// header(i): jump latch
/// latch:     one = 1; i1 = i + one; c = i1 < n; jumpif c header(i1) else exit
/// exit:      return i
/// ```
/// The latch passes `i1` to the header's `i`, and exits to a block that
/// reads the header's `i`: `i` is live at the latch's end through the exit
/// only, and the edge into the header kills it only on its own way.
#[test]
fn liveness_keeps_a_header_parameter_live_through_the_exit_of_a_latch() {
    let mut f = fixture();
    let (nn, int, bool) = (f.interner.intern("n"), f.int, f.bool);
    let mut builder = f.module.build_body(false, int);
    let n = builder.param(nn, int, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let i0 = builder.push(constant(0), int, Span::ZERO);
    let header = builder.new_block(&[int]);
    let i = builder.block_params(header)[0];
    let [latch, exit] = [(); 2].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header, vec![i0]), acvus_ast::Span::ZERO);
    builder.switch_to(header);
    builder.terminate(jump(latch, Vec::new()), acvus_ast::Span::ZERO);
    builder.switch_to(latch);
    let one = builder.push(constant(1), int, Span::ZERO);
    let i1 = builder.push(add(i, one), int, Span::ZERO);
    let c = builder.push(binary(BinOp::Lt, i1, n), bool, Span::ZERO);
    builder.terminate(Terminator::JumpIf { cond: c, then_to: header, then_args: vec![i1], else_to: exit, else_args: Vec::new() }, acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    builder.terminate(ret(i), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let live = facts_of::<super::Liveness>(&f, id);
    let set = |values: &[ValueId]| values.iter().copied().collect::<rustc_hash::FxHashSet<ValueId>>();
    assert_eq!(*live.live_at_end(latch), set(&[i, i1, n]), "i through the exit, i1 on the edge into the header");
    assert_eq!(*live.live_at_start(latch), set(&[i, n]));
    assert_eq!(*live.live_at_start(header), set(&[i, n]));
    assert_eq!(*live.live_at_end(entry), set(&[i0, n]), "the header's i is defined on the edge, not before it");
    assert_eq!(*live.live_at_start(exit), set(&[i]));
    assert_eq!(*live.live_at_end(exit), set(&[]));
}

// -- Content liveness ------------------------------------------------------------

/// ```text
/// entry:  a = 1; s = a; b = 2; s = b; r = &s; f(r); return b
/// ```
/// The first content of `s` is replaced before any read, so it is dead from
/// its store; the second is read through `r` by the call, and dead after it.
#[test]
fn content_liveness_ends_at_a_whole_store_and_follows_a_read_through_a_reference() {
    let mut f = loan_fixture();
    let (int, ref_int, passes_nothing) = (f.int, f.ref_int, f.passes_nothing);
    let callee = Callee::Direct(QualifiedRef::root(f.interner.intern("f")));
    let mut builder = f.module.build_body(false, int);
    let entry = builder.current_block();
    let s = builder.slot(int, Span::ZERO);
    let a = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(Op::Assign { target: RefTarget::Var(s), path: Vec::new(), value: a, restores: false }, Span::ZERO);
    let b = builder.push(constant(2), int, Span::ZERO);
    builder.push_unit(Op::Assign { target: RefTarget::Var(s), path: Vec::new(), value: b, restores: false }, Span::ZERO);
    let r = builder.push(reference(s, Mutability::Shared), ref_int, Span::ZERO);
    builder.push(Op::FunctionCall { callee, callee_ty: passes_nothing, args: vec![r] }, ref_int, Span::ZERO);
    builder.terminate(ret(b), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let content = f.facts::<ContentLiveness>(id);
    let live_after = |index: usize| content.live_after(site_at(&f.module, id, entry, index)).contains(&s);
    assert!(!live_after(1), "the first content is replaced before any read");
    assert!(live_after(3), "the second content is read through r");
    assert!(live_after(4));
    assert!(!live_after(5), "nothing reads s after the call");
    assert!(!content.live_at_start(entry).contains(&s));
    assert!(content.live_at_end(entry).is_empty(), "the body's end reads none of its own slots");
}

// -- Loans and escapes ---------------------------------------------------------

struct LoanFixture {
    module: Module,
    interner: Interner,
    int: Ty,
    bool: Ty,
    /// `&i64`.
    ref_int: Ty,
    /// `&mut i64`.
    mut_int: Ty,
    /// `Option<&i64>`.
    option_ref: Ty,
    /// `(&i64, &i64)`.
    pair_of_refs: Ty,
    /// `&(&i64, &i64)` and `&mut (&i64, &i64)`.
    ref_pair: Ty,
    mut_pair: Ty,
    /// A pure `fn(&i64) -> &i64` whose result takes its argument's loans.
    passes_through: Ty,
    /// A pure `fn(&i64) -> &i64` whose result takes no loan.
    passes_nothing: Ty,
    /// A pure `fn(i64) -> i64`.
    takes_int: Ty,
}

fn loan_fixture() -> LoanFixture {
    let mut types = Types::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let bool = types.intern(TyKind::Bool);
    let ref_int = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    let mut_int = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(int)));
    let option_ref = types.intern(TyKind::Option(ref_int, ()));
    let pair_of_refs = types.intern(TyKind::Tuple(Box::new([ref_int, ref_int]), ()));
    let ref_pair = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(pair_of_refs)));
    let mut_pair = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(pair_of_refs)));
    let mut function = |params: &[Ty], ret: Ty, flows: Flows| {
        types.intern(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(flows),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        })
    };
    let through = Flows::of([Flow { to: FlowEnd::Result, from: FlowEnd::Param(0), alignment: Alignment::Aligned }]);
    let passes_through = function(&[ref_int], ref_int, through);
    let passes_nothing = function(&[ref_int], ref_int, Flows::none());
    let takes_int = function(&[int], int, Flows::none());
    LoanFixture {
        module: Module::new(types),
        interner: Interner::new(),
        int,
        bool,
        ref_int,
        mut_int,
        option_ref,
        pair_of_refs,
        ref_pair,
        mut_pair,
        passes_through,
        passes_nothing,
        takes_int,
    }
}

impl LoanFixture {
    fn facts<A>(&self, body: BodyId) -> Rc<A::Facts>
    where
        A: Analysis,
    {
        let decls = empty_declarations(&self.interner);
        let mut am = AnalysisManager::new(&decls);
        am.get::<A>(&self.module, body)
    }

    fn call(&self, callee_ty: Ty, args: Vec<ValueId>) -> Op {
        Op::FunctionCall { callee: Callee::Direct(QualifiedRef::root(self.interner.intern("f"))), callee_ty, args }
    }
}

fn reference(slot: ValueId, mutability: Mutability) -> Op {
    Op::Ref { target: RefTarget::Var(slot), path: Vec::new(), mutability }
}

fn reborrow(of: ValueId, mutability: Mutability) -> Op {
    Op::Ref { target: RefTarget::Through(of), path: Vec::new(), mutability }
}

fn assign(slot: ValueId, value: ValueId) -> Op {
    Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }
}

fn take(slot: ValueId) -> Op {
    Op::Take { target: RefTarget::Var(slot), path: Vec::new(), taken_out: false }
}

fn loan(storage: Storage, mode: Mode) -> Loan {
    Loan { storage, path: Vec::new(), mode }
}

fn loans_of(loans: &Loans, value: ValueId, position: usize) -> Vec<Loan> {
    loans.of(value, position).iter().cloned().collect()
}

/// `x = 1; r = &x; m = &mut x; s = &*r; return 0`.
#[test]
fn a_ref_names_its_slot_and_a_reborrow_names_what_its_source_names() {
    let mut f = loan_fixture();
    let (int, ref_int, mut_int) = (f.int, f.ref_int, f.mut_int);
    let mut builder = f.module.build_body(false, int);
    let x = builder.slot(int, Span::ZERO);
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    let r = builder.push(reference(x, Mutability::Shared), ref_int, Span::ZERO);
    let m = builder.push(reference(x, Mutability::Mut), mut_int, Span::ZERO);
    let s = builder.push(reborrow(m, Mutability::Shared), ref_int, Span::ZERO);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let loans = f.facts::<Loans>(id);
    assert_eq!(loans_of(&loans, r, 0), [loan(Storage::Slot(x), Mode::Shared)]);
    assert_eq!(loans_of(&loans, m, 0), [loan(Storage::Slot(x), Mode::Exclusive)]);
    assert_eq!(loans_of(&loans, s, 0), [loan(Storage::Slot(x), Mode::Shared)], "a shared reborrow of the &mut");
    assert_eq!(loans.positions_of(one), 0);
    let entry = f.module.body(id).entry();
    assert_eq!(*loans.writes(site_at(&f.module, id, entry, 1)), BTreeSet::from([Storage::Slot(x)]));
    assert!(loans.reads(site_at(&f.module, id, entry, 2)).is_empty(), "a Ref names its place and reads nothing");
}

/// ```text
/// a = 1; b = 2; ra = &a; rb = &b
/// y = ra; t1 = y; y = rb; t2 = y; return 0
/// ```
/// A whole-slot `Assign` replaces what the slot held, so each `Take` sees
/// only the reference stored last.
#[test]
fn a_reference_stored_into_a_slot_is_taken_back_out_as_stored_last() {
    let mut f = loan_fixture();
    let (int, ref_int) = (f.int, f.ref_int);
    let mut builder = f.module.build_body(false, int);
    let a = builder.slot(int, Span::ZERO);
    let b = builder.slot(int, Span::ZERO);
    let y = builder.slot(ref_int, Span::ZERO);
    let entry = builder.current_block();
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(a, one), Span::ZERO);
    builder.push_unit(assign(b, one), Span::ZERO);
    let ra = builder.push(reference(a, Mutability::Shared), ref_int, Span::ZERO);
    let rb = builder.push(reference(b, Mutability::Shared), ref_int, Span::ZERO);
    builder.push_unit(assign(y, ra), Span::ZERO);
    let t1 = builder.push(take(y), ref_int, Span::ZERO);
    builder.push_unit(assign(y, rb), Span::ZERO);
    let t2 = builder.push(take(y), ref_int, Span::ZERO);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let loans = f.facts::<Loans>(id);
    assert_eq!(loans_of(&loans, t1, 0), [loan(Storage::Slot(a), Mode::Shared)]);
    assert_eq!(loans_of(&loans, t2, 0), [loan(Storage::Slot(b), Mode::Shared)]);
    let before_t2 = loans.content_before(site_at(&f.module, id, entry, 8), Storage::Slot(y));
    assert_eq!(before_t2, [BTreeSet::from([loan(Storage::Slot(b), Mode::Shared)])]);
    assert_eq!(*loans.reads(site_at(&f.module, id, entry, 8)), BTreeSet::from([Storage::Slot(y)]));
}

/// `x = 1; r = &x; p = through(r); q = nothing(r); return 0`.
#[test]
fn a_call_moves_loans_to_its_result_only_as_its_flows_state() {
    let mut f = loan_fixture();
    let (int, ref_int) = (f.int, f.ref_int);
    let (through, nothing) = (f.call(f.passes_through, Vec::new()), f.call(f.passes_nothing, Vec::new()));
    let mut builder = f.module.build_body(false, int);
    let x = builder.slot(int, Span::ZERO);
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    let r = builder.push(reference(x, Mutability::Shared), ref_int, Span::ZERO);
    let with_arg = |op: Op| match op {
        Op::FunctionCall { callee, callee_ty, .. } => Op::FunctionCall { callee, callee_ty, args: vec![r] },
        other => other,
    };
    let p = builder.push(with_arg(through), ref_int, Span::ZERO);
    let q = builder.push(with_arg(nothing), ref_int, Span::ZERO);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let loans = f.facts::<Loans>(id);
    assert_eq!(loans_of(&loans, p, 0), [loan(Storage::Slot(x), Mode::Shared)]);
    assert!(loans.of(q, 0).is_empty(), "no flow reaches the result");
}

/// ```text
/// entry(c): a = 1; b = 1; diamond c { then } else { else } join
/// then:     ra = &a; jump join(ra)
/// else:     rb = &b; jump join(rb)
/// join(p):  return 0
/// ```
#[test]
fn a_block_parameter_joins_the_references_its_edges_pass() {
    let mut f = loan_fixture();
    let (nc, int, bool, ref_int) = (f.interner.intern("c"), f.int, f.bool, f.ref_int);
    let mut builder = f.module.build_body(false, int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let a = builder.slot(int, Span::ZERO);
    let b = builder.slot(int, Span::ZERO);
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(a, one), Span::ZERO);
    builder.push_unit(assign(b, one), Span::ZERO);
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[ref_int]);
    let p = builder.block_params(join)[0];
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    let ra = builder.push(reference(a, Mutability::Shared), ref_int, Span::ZERO);
    builder.terminate(jump(join, vec![ra]), acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    let rb = builder.push(reference(b, Mutability::Shared), ref_int, Span::ZERO);
    builder.terminate(jump(join, vec![rb]), acvus_ast::Span::ZERO);
    builder.switch_to(join);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let loans = f.facts::<Loans>(id);
    assert_eq!(
        loans_of(&loans, p, 0),
        [loan(Storage::Slot(a), Mode::Shared), loan(Storage::Slot(b), Mode::Shared)]
    );
}

/// `entry(o: Option<&i64>): u = unwrap o; v = *u; return v`: the parameter's
/// one position names its own outside storage, and so does the payload.
#[test]
fn an_option_of_a_reference_parameter_names_its_outside_storage() {
    let mut f = loan_fixture();
    let (no, int, ref_int, option_ref) = (f.interner.intern("o"), f.int, f.ref_int, f.option_ref);
    let mut builder = f.module.build_body(false, int);
    let o = builder.param(no, option_ref, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let u = builder.push(Op::UnwrapVariant { src: o }, ref_int, Span::ZERO);
    let v = builder.push(Op::Take { target: RefTarget::Through(u), path: Vec::new(), taken_out: false }, int, Span::ZERO);
    builder.terminate(ret(v), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let loans = f.facts::<Loans>(id);
    let outside = Storage::Outside { of: o, position: 0 };
    assert_eq!(loans_of(&loans, o, 0), [loan(outside, Mode::Shared)]);
    assert_eq!(loans_of(&loans, u, 0), [loan(outside, Mode::Shared)]);
    assert_eq!(*loans.reads(site_at(&f.module, id, entry, 1)), BTreeSet::from([outside]));
}

fn loan_at(storage: Storage, path: Vec<PathSeg>, mode: Mode) -> Loan {
    Loan { storage, path, mode }
}

/// A pair `p: (&i64, &i64)` of `&a` and `&b`, stored whole into its slot;
/// the rest of the body is `rest`'s, given `p`, `a` and `b`.
fn with_pair_of_refs<R>(f: &mut LoanFixture, rest: impl FnOnce(&mut crate::ir::cfg::Builder<'_>, [ValueId; 3]) -> R) -> (BodyId, R) {
    let (int, ref_int, pair_of_refs) = (f.int, f.ref_int, f.pair_of_refs);
    let mut builder = f.module.build_body(false, int);
    let p = builder.slot(pair_of_refs, Span::ZERO);
    let a = builder.slot(int, Span::ZERO);
    let b = builder.slot(int, Span::ZERO);
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(a, one), Span::ZERO);
    builder.push_unit(assign(b, one), Span::ZERO);
    let ra = builder.push(reference(a, Mutability::Shared), ref_int, Span::ZERO);
    let rb = builder.push(reference(b, Mutability::Shared), ref_int, Span::ZERO);
    let pair = builder.push(Op::MakeTuple { elements: vec![ra, rb] }, pair_of_refs, Span::ZERO);
    builder.push_unit(assign(p, pair), Span::ZERO);
    let made = rest(&mut builder, [p, a, b]);
    let zero = builder.push(constant(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    (builder.seal().expect("the body has no open state"), made)
}

/// Lend: a `Ref` of a part by name makes one loan on that part, at the
/// reference's mode.
#[test]
fn a_ref_of_a_part_by_name_lends_that_part() {
    let mut f = loan_fixture();
    let (ref_int, mut_int) = (f.ref_int, f.mut_int);
    let (id, [shared, exclusive, p]) = with_pair_of_refs(&mut f, |builder, [p, _, _]| {
        let at = |index, mutability| Op::Ref { target: RefTarget::Var(p), path: vec![PathSeg::Index(index)], mutability };
        let shared = builder.push(at(0, Mutability::Shared), ref_int, Span::ZERO);
        let exclusive = builder.push(at(1, Mutability::Mut), mut_int, Span::ZERO);
        [shared, exclusive, p]
    });

    let loans = f.facts::<Loans>(id);
    let on = |index| vec![PathSeg::Index(index)];
    assert_eq!(loans_of(&loans, shared, 0), [loan_at(Storage::Slot(p), on(0), Mode::Shared)]);
    assert_eq!(loans_of(&loans, exclusive, 0), [loan_at(Storage::Slot(p), on(1), Mode::Exclusive)]);
}

/// Reborrow: a `Ref` through `m` makes no loan; it holds `m`'s, the path
/// extended by its own, at the weaker mode. Through a shared reference a
/// `&mut`'s loan is read shared.
#[test]
fn a_reborrow_extends_its_sources_loans_by_its_path_at_the_weaker_mode() {
    let mut f = loan_fixture();
    let (ref_int, mut_pair, ref_pair) = (f.ref_int, f.mut_pair, f.ref_pair);
    let (id, [m, part, shared, through_shared, p]) = with_pair_of_refs(&mut f, |builder, [p, _, _]| {
        let m = builder.push(reference(p, Mutability::Mut), mut_pair, Span::ZERO);
        let through = |of, mutability| Op::Ref { target: RefTarget::Through(of), path: vec![PathSeg::Index(1)], mutability };
        let part = builder.push(through(m, Mutability::Shared), ref_int, Span::ZERO);
        let shared = builder.push(reborrow(m, Mutability::Shared), ref_pair, Span::ZERO);
        let through_shared = builder.push(through(shared, Mutability::Shared), ref_int, Span::ZERO);
        [m, part, shared, through_shared, p]
    });

    let loans = f.facts::<Loans>(id);
    assert_eq!(loans_of(&loans, m, 0), [loan_at(Storage::Slot(p), Vec::new(), Mode::Exclusive)]);
    assert_eq!(loans_of(&loans, part, 0), [loan_at(Storage::Slot(p), vec![PathSeg::Index(1)], Mode::Shared)]);
    assert_eq!(loans_of(&loans, shared, 0), [loan_at(Storage::Slot(p), Vec::new(), Mode::Shared)]);
    assert_eq!(loans_of(&loans, through_shared, 0), [loan_at(Storage::Slot(p), vec![PathSeg::Index(1)], Mode::Shared)]);
}

/// Move: an `Assign` gives the content at its path's positions what the
/// value holds, and a `Take` gives its value what the content holds at the
/// positions its path reaches.
#[test]
fn a_move_gives_what_the_content_holds_at_its_paths_positions() {
    let mut f = loan_fixture();
    let (ref_int, mut_pair) = (f.ref_int, f.mut_pair);
    let (id, [first, second, a]) = with_pair_of_refs(&mut f, |builder, [p, a, _]| {
        let take_at = |index| Op::Take { target: RefTarget::Var(p), path: vec![PathSeg::Index(index)], taken_out: false };
        let m = builder.push(reference(p, Mutability::Mut), mut_pair, Span::ZERO);
        let ra = builder.push(reference(a, Mutability::Shared), ref_int, Span::ZERO);
        let store = Op::Assign { target: RefTarget::Through(m), path: vec![PathSeg::Index(1)], value: ra, restores: false };
        builder.push_unit(store, Span::ZERO);
        let first = builder.push(take_at(0), ref_int, Span::ZERO);
        let second = builder.push(take_at(1), ref_int, Span::ZERO);
        [first, second, a]
    });

    let loans = f.facts::<Loans>(id);
    assert_eq!(loans_of(&loans, first, 0), [loan(Storage::Slot(a), Mode::Shared)], "the first part still holds `&a`");
    assert_eq!(loans_of(&loans, second, 0), [loan(Storage::Slot(a), Mode::Shared)], "the store through `&mut p` replaced `&b`");
}

/// A `Take` of a `&mut` is a reborrow: the place keeps what it held.
#[test]
fn a_take_of_a_mut_reference_leaves_its_place_holding_it() {
    let mut f = loan_fixture();
    let (int, mut_int) = (f.int, f.mut_int);
    let mut builder = f.module.build_body(false, int);
    let x = builder.slot(int, Span::ZERO);
    let m = builder.slot(mut_int, Span::ZERO);
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    let rx = builder.push(reference(x, Mutability::Mut), mut_int, Span::ZERO);
    builder.push_unit(assign(m, rx), Span::ZERO);
    let taken = builder.push(take(m), mut_int, Span::ZERO);
    let again = builder.push(take(m), mut_int, Span::ZERO);
    builder.terminate(ret(one), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let loans = f.facts::<Loans>(id);
    assert_eq!(loans_of(&loans, taken, 0), [loan(Storage::Slot(x), Mode::Exclusive)]);
    assert_eq!(loans_of(&loans, again, 0), [loan(Storage::Slot(x), Mode::Exclusive)]);
}

/// Part: reading a part of a value gives what that part's positions hold.
#[test]
fn a_part_of_a_value_holds_what_its_positions_hold() {
    let mut f = loan_fixture();
    let (ref_int, pair_of_refs) = (f.ref_int, f.pair_of_refs);
    let (id, [second, b]) = with_pair_of_refs(&mut f, |builder, [p, _, b]| {
        let pair = builder.push(take(p), pair_of_refs, Span::ZERO);
        let second = builder.push(Op::TupleIndex { tuple: pair, index: 1 }, ref_int, Span::ZERO);
        [second, b]
    });

    let loans = f.facts::<Loans>(id);
    assert_eq!(loans_of(&loans, second, 0), [loan(Storage::Slot(b), Mode::Shared)]);
}

/// `x = 1; t = x; return t`: only whole-slot uses.
#[test]
fn a_slot_used_only_whole_escapes_nothing() {
    let mut f = loan_fixture();
    let int = f.int;
    let mut builder = f.module.build_body(false, int);
    let x = builder.slot(int, Span::ZERO);
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    let t = builder.push(take(x), int, Span::ZERO);
    builder.terminate(ret(t), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    assert_eq!(f.facts::<Escapes>(id).escape(x), None);
}

/// `x = 1; t = x; r = &x; return t`.
#[test]
fn a_ref_of_a_slot_is_where_it_escapes() {
    let mut f = loan_fixture();
    let (int, ref_int) = (f.int, f.ref_int);
    let mut builder = f.module.build_body(false, int);
    let x = builder.slot(int, Span::ZERO);
    let entry = builder.current_block();
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    let t = builder.push(take(x), int, Span::ZERO);
    builder.push(reference(x, Mutability::Shared), ref_int, Span::ZERO);
    builder.terminate(ret(t), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    assert_eq!(f.facts::<Escapes>(id).escape(x), Some(site_at(&f.module, id, entry, 3)));
}

/// `x = 1; return x`: the terminator names the slot.
#[test]
fn a_returned_slot_escapes_at_the_return() {
    let mut f = loan_fixture();
    let int = f.int;
    let mut builder = f.module.build_body(false, int);
    let x = builder.slot(int, Span::ZERO);
    let entry = builder.current_block();
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    builder.terminate(ret(x), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    assert_eq!(f.facts::<Escapes>(id).escape(x), Some(Site::Term(entry)));
}

/// `x = 1; t = f(x); return t`: the slot is a call's argument.
#[test]
fn a_slot_passed_to_a_call_escapes_at_the_call() {
    let mut f = loan_fixture();
    let int = f.int;
    let mut builder = f.module.build_body(false, int);
    let x = builder.slot(int, Span::ZERO);
    let entry = builder.current_block();
    let one = builder.push(constant(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    let callee = Callee::Direct(QualifiedRef::root(f.interner.intern("f")));
    let t = builder.push(Op::FunctionCall { callee, callee_ty: f.takes_int, args: vec![x] }, int, Span::ZERO);
    builder.terminate(ret(t), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    assert_eq!(f.facts::<Escapes>(id).escape(x), Some(site_at(&f.module, id, entry, 2)));
}
