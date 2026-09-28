//! The editor's operations, their refusals, the needs it keeps current, its
//! debug checks and its invariants' checkers. Bodies are built by the
//! Builder; facts come from test-only implementations: dominance computed
//! for the body at each refresh, order and loans answered from fixed tables.

use std::fmt::Write;

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, QualifiedRef};
use rustc_hash::{FxHashMap, FxHashSet};

use super::dominators::{DefPoint, Dominators, def_points};
use super::*;
use crate::ir::cfg::{Aspect, Builder, Inst, OrderUse, Stages, Terminator};
use crate::ir::{BinOp, Callee, Checked, Overflow, RefTarget, UnaryOp};
use crate::ir::{ForSource, SwitchKey};
use crate::ty::{
    Calls, Concrete, Contexts, Cost, Effect as CallEffect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability,
    ObjectTy, Reissue, TyKind, TypeArg, Types,
};

// -- Fixture -------------------------------------------------------------

struct Fixture {
    module: Module,
    interner: Interner,
    int: Ty,
    bool: Ty,
    /// An opaque callee type that does not let a call be split.
    opaque: Ty,
    /// An opaque callee type that lets a call be split.
    splittable: Ty,
    /// A pure callee type.
    pure: Ty,
    /// A pure callee type that lets a call be split.
    pure_splittable: Ty,
    /// A pure callee type that writes a context.
    pure_writing: Ty,
    /// An idempotent callee type that writes no context.
    idempotent: Ty,
    /// `&str`.
    str_ref: Ty,
    /// `Result<i64, bool>`.
    result: Ty,
    /// A declared struct with one field `x: i64`.
    object: Ty,
}

fn fixture() -> Fixture {
    let interner = Interner::new();
    let mut types = Types::<Concrete>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let bool = types.intern(TyKind::Bool);
    let mut call_ty = |effect: CallEffect| {
        types.intern(TyKind::Fn {
            params: Box::new([]),
            ret: int,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        })
    };
    let split = SplitCost::WithCost(Cost::stated(1));
    let opaque = call_ty(CallEffect::opaque());
    let splittable = call_ty(CallEffect::new(Reissue::Opaque, false, Contexts::new(), Contexts::new(), split));
    let pure = call_ty(CallEffect::pure());
    let pure_splittable = call_ty(CallEffect::new(Reissue::Pure, true, Contexts::new(), Contexts::new(), split));
    let written = Contexts::from([QualifiedRef::root(interner.intern("page"))]);
    let pure_writing = call_ty(CallEffect::new(Reissue::Pure, false, Contexts::new(), written, SplitCost::Unavailable));
    let idempotent =
        call_ty(CallEffect::new(Reissue::Idempotent, false, Contexts::new(), Contexts::new(), SplitCost::Unavailable));
    let str_ty = types.intern(TyKind::Str);
    let str_ref = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(str_ty)));
    let result = types.intern(TyKind::Result(int, bool));
    let point = QualifiedRef::root(interner.intern("Point"));
    let object = types.intern(TyKind::Object(ObjectTy::Declared(point, Box::new([(interner.intern("x"), int)]))));
    Fixture {
        module: Module::new(types, Flows::none()),
        interner,
        int,
        bool,
        opaque,
        splittable,
        pure,
        pure_splittable,
        pure_writing,
        idempotent,
        str_ref,
        result,
        object,
    }
}

impl Fixture {
    fn name(&self, name: &str) -> acvus_utils::Astr {
        self.interner.intern(name)
    }

    fn call(&self, callee_ty: Ty) -> Op {
        Op::FunctionCall { callee: Callee::Direct(QualifiedRef::root(self.name("f"))), callee_ty, args: Vec::new() }
    }

    fn body(&self, body: BodyId) -> &Body {
        self.module.body(body)
    }

    fn build(&mut self, ordered: bool, ret: Ty) -> Builder<'_> {
        self.module.build_body(ordered, ret)
    }

    fn insts(&self, body: BodyId, block: BlockId) -> &[Inst] {
        self.body(body).block(block).insts()
    }
}

fn at(n: u32) -> Span {
    Span::new(n as usize, n as usize + 1)
}

fn ret(value: ValueId) -> Terminator {
    Terminator::Return { value, order: None, span: Span::ZERO }
}

fn jump(to: BlockId) -> Terminator {
    Terminator::Jump { to, args: Vec::new() }
}

fn diamond(cond: ValueId, then_to: BlockId, else_to: BlockId, join: BlockId) -> Terminator {
    Terminator::Diamond { cond, then_to, then_args: Vec::new(), else_to, else_args: Vec::new(), join }
}

fn add(left: ValueId, right: ValueId) -> Op {
    Op::BinOp { op: BinOp::Add(Overflow::Trap), left, right }
}

fn neg(operand: ValueId) -> Op {
    Op::UnaryOp { op: UnaryOp::Neg(Overflow::Trap), operand }
}

fn check_add(left: ValueId, right: ValueId) -> Op {
    Op::Check { op: Checked::Add, left, right }
}

fn int_const(value: i128) -> Op {
    Op::Const { value: Literal::Int(value) }
}

fn inst_ref(block: BlockId, index: usize) -> InstRef {
    InstRef { block, index }
}

fn entry_of(f: &Fixture, body: BodyId) -> BlockId {
    f.body(body).entry()
}

fn result_of(inst: &Inst) -> ValueId {
    inst.result().expect("the instruction yields a value")
}

fn edge_of(inst: &Inst) -> crate::ir::OrderEdge {
    let Some(OrderUse::Edge(edge)) = inst.order() else {
        panic!("{inst:?} is an effectful call")
    };
    edge
}

fn returned_order(body: &Body, block: BlockId) -> Option<ValueId> {
    let Terminator::Return { order, .. } = body.block(block).terminator() else {
        panic!("{block:?} returns")
    };
    *order
}

/// Everything a refused edit must leave as it was: every block's
/// parameters, instructions and terminator, and every value's type and use
/// count.
fn snapshot(body: &Body) -> String {
    let mut out = String::new();
    for (id, block) in body.blocks() {
        writeln!(out, "{id:?} {:?}", block.params()).expect("a String takes every write");
        for inst in block.insts() {
            writeln!(out, "  {inst:?}").expect("a String takes every write");
        }
        writeln!(out, "  {:?}", block.terminator()).expect("a String takes every write");
    }
    let mut values: Vec<ValueId> = body.values().collect();
    values.sort();
    for value in values {
        writeln!(out, "{value:?}: {:?} used {}", body.ty_of(value), body.use_count(value)).expect("a String takes every write");
    }
    out
}

// -- Facts for tests -----------------------------------------------------

/// Dominance computed from the body at each refresh; order and loans from
/// fixed tables; and a record of every refresh.
#[derive(Default)]
struct TestFacts {
    dominance: Option<Dominance>,
    refreshed: Vec<Vec<Aspect>>,
    kept_in_order: FxHashSet<(InstRef, InstRef)>,
    conflicting: FxHashSet<(InstRef, InstRef)>,
    escaping: FxHashMap<ValueId, InstRef>,
}

impl TestFacts {
    fn of(module: &Module, body: BodyId) -> TestFacts {
        let mut facts = TestFacts::default();
        facts.compute(module, body);
        facts
    }

    fn compute(&mut self, module: &Module, body: BodyId) {
        let body = module.body(body);
        let successors = body
            .blocks()
            .map(|(_, block)| block.terminator().edges().iter().map(|edge| edge.to).collect())
            .collect();
        self.dominance = Some(Dominance { dominators: Dominators::of(body), points: def_points(body), successors });
    }
}

/// Dominance of one body, and its edges for post-dominance, found by
/// search: `below` post-dominates `above` when no exit is reached from
/// `above` without passing `below`.
struct Dominance {
    dominators: Dominators,
    points: FxHashMap<ValueId, DefPoint>,
    successors: Vec<Vec<BlockId>>,
}

impl Dominance {
    fn exits_avoiding(&self, from: BlockId, avoided: BlockId) -> bool {
        let mut seen = FxHashSet::default();
        let mut pending = vec![from];
        while let Some(block) = pending.pop() {
            if block == avoided || !seen.insert(block) {
                continue;
            }
            if self.successors[block.index()].is_empty() {
                return true;
            }
            pending.extend(self.successors[block.index()].iter().copied());
        }
        false
    }
}

impl Facts for TestFacts {
    fn refresh(&mut self, module: &Module, body: BodyId, changes: &Changes) {
        self.compute(module, body);
        self.refreshed.push(changes.aspects().collect());
    }
}

impl DominanceFacts for TestFacts {
    fn dominates(&self, value: ValueId, at: Position) -> bool {
        let dominance = self.dominance.as_ref().expect("the facts are computed before they are read");
        dominance.dominators.dominates_site(dominance.points[&value], position_site(at))
    }

    fn position_dominates(&self, at: Position, site: Site) -> bool {
        let dominators = &self.dominance.as_ref().expect("the facts are computed before they are read").dominators;
        let (block, index) = match at {
            Position::Before(inst) => (inst.block, Some(inst.index)),
            Position::End(block) => (block, None),
        };
        match site {
            Site::Inst(at, at_index) if at == block => index.is_some_and(|index| index <= at_index),
            Site::Term(at) if at == block => true,
            Site::Inst(at, _) | Site::Term(at) => dominators.dominates_block(block, at),
        }
    }

    fn control_equivalent(&self, a: Position, b: Position) -> bool {
        let dominance = self.dominance.as_ref().expect("the facts are computed before they are read");
        let (a, b) = (position_block(a), position_block(b));
        let dominates = |above, below| dominance.dominators.dominates_block(above, below);
        let post_dominates = |below: BlockId, above: BlockId| !dominance.exits_avoiding(above, below);
        a == b || (dominates(a, b) && post_dominates(b, a)) || (dominates(b, a) && post_dominates(a, b))
    }
}

fn position_site(at: Position) -> Site {
    match at {
        Position::Before(inst) => Site::Inst(inst.block, inst.index),
        Position::End(block) => Site::Term(block),
    }
}

fn position_block(at: Position) -> BlockId {
    match at {
        Position::Before(inst) => inst.block,
        Position::End(block) => block,
    }
}

impl OrderFacts for TestFacts {
    fn may_reorder(&self, a: InstRef, b: InstRef) -> bool {
        !self.kept_in_order.contains(&(a, b))
    }
}

impl LoanFacts for TestFacts {
    fn conflicts(&self, inst: InstRef, write: InstRef) -> bool {
        self.conflicting.contains(&(inst, write))
    }

    fn escapes(&self, slot: ValueId) -> Option<InstRef> {
        self.escaping.get(&slot).copied()
    }
}

// -- Bodies ----------------------------------------------------------------

/// One block: `x = a + a; y = -x; z = a + a; w = z + x; return y`.
fn straight_line(f: &mut Fixture) -> (BodyId, [ValueId; 4]) {
    let (name, int) = (f.name("a"), f.int);
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    let x = builder.push(add(a, a), int, at(1));
    let y = builder.push(neg(x), int, at(2));
    let z = builder.push(add(a, a), int, at(3));
    builder.push(add(z, x), int, at(4));
    builder.terminate(ret(y));
    (builder.seal().expect("the body has no open state"), [a, x, y, z])
}

/// An ordered body of one block: `r1 = f(); r2 = g(); check(a + a); return a`.
fn two_calls(f: &mut Fixture, callee: Ty) -> (BodyId, ValueId) {
    let (name, int) = (f.name("a"), f.int);
    let (first, second) = (f.call(callee), f.call(callee));
    let mut builder = f.build(true, f.int);
    let a = builder.param(name, int);
    builder.push_effect(first, int, at(1));
    builder.push_effect(second, int, at(2));
    builder.push_unit(check_add(a, a), at(3));
    builder.terminate(ret(a));
    (builder.seal().expect("the body has no open state"), a)
}

// -- replace_uses ------------------------------------------------------------

#[test]
fn replace_uses_moves_every_use_to_a_dominating_value_and_refreshes_the_needs() {
    let mut f = fixture();
    let (id, [a, x, _, z]) = straight_line(&mut f);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    let replaced = editor.replace_uses(z, x);
    let changes = editor.finish();

    assert_eq!(replaced, Ok(()));
    assert!(changes.touched(Aspect::Instructions));
    assert_eq!(facts.refreshed, vec![vec![Aspect::Instructions]]);
    let body = f.body(id);
    assert_eq!((body.use_count(z), body.use_count(x)), (0, 3));
    assert_eq!(body.use_count(a), 4);
}

#[test]
fn replace_uses_refuses_a_value_that_does_not_dominate_a_use_and_leaves_the_body() {
    let mut f = fixture();
    let (id, [_, x, _, z]) = straight_line(&mut f);
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    let refused = editor.replace_uses(x, z);
    editor.finish();

    assert_eq!(refused, Err(DoesNotDominate { use_at: Site::Inst(entry, 1) }));
    assert_eq!(snapshot(f.body(id)), before);
    assert!(facts.refreshed.is_empty(), "a refused edit changes nothing to refresh");
}

#[test]
#[should_panic(expected = "is replaced only by a value of its type")]
fn replace_uses_of_another_type_is_a_callers_defect() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    let b = builder.push(Op::Const { value: Literal::Bool(true) }, bool, at(1));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    let _ = editor.replace_uses(a, b);
}

// -- insert ----------------------------------------------------------------

#[test]
fn insert_puts_a_pure_instruction_at_its_position() {
    let mut f = fixture();
    let (id, [a, x, _, _]) = straight_line(&mut f);
    let (entry, int) = (entry_of(&f, id), f.int);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    let doubled = editor.insert(Position::Before(inst_ref(entry, 1)), add(x, x), int, at(9));
    let at_end = editor.insert(Position::End(entry), neg(a), int, at(10));
    editor.finish();

    let insts = f.insts(id, entry);
    assert_eq!(insts.len(), 6);
    assert_eq!((result_of(&insts[1]), insts[1].span()), (doubled, at(9)));
    assert_eq!(result_of(&insts[5]), at_end);
    assert_eq!(facts.refreshed.len(), 2, "each edit refreshes the needs");
}

#[test]
#[should_panic(expected = "takes an order token")]
fn inserting_an_effectful_call_is_a_callers_defect() {
    let mut f = fixture();
    let (id, _) = straight_line(&mut f);
    let (entry, int, call) = (entry_of(&f, id), f.int, f.call(f.opaque));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    editor.insert(Position::End(entry), call, int, at(9));
}

#[test]
#[should_panic(expected = "is written only by the core")]
fn inserting_a_merge_is_a_callers_defect() {
    let mut f = fixture();
    let (id, [a, ..]) = straight_line(&mut f);
    let (entry, int) = (entry_of(&f, id), f.int);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    editor.insert(Position::End(entry), Op::Merge { orders: vec![a] }, int, at(9));
}

// -- remove ----------------------------------------------------------------

#[test]
fn remove_takes_out_an_unused_pure_instruction_and_a_pure_call() {
    let mut f = fixture();
    let (name, int, pure) = (f.name("a"), f.int, f.call(f.pure));
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    builder.push(pure, int, at(1));
    builder.push(min(a, a), int, at(2));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);

    let mut facts = NoFacts;
    let mut editor = Editor::<Remove>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.remove(inst_ref(entry, 1)), Ok(()));
    assert_eq!(editor.remove(inst_ref(entry, 0)), Ok(()));
    editor.finish();
    assert!(f.insts(id, entry).is_empty());
}

#[test]
fn remove_refuses_an_instruction_whose_result_is_used() {
    let mut f = fixture();
    let (id, _) = straight_line(&mut f);
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = NoFacts;
    let mut editor = Editor::<Remove>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.remove(inst_ref(entry, 0)), Err(Unremovable::StillUsed));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

#[test]
fn remove_refuses_an_opaque_call_and_a_call_that_writes() {
    let mut f = fixture();
    let (id, _) = { let callee = f.opaque; two_calls(&mut f, callee) };
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = NoFacts;
    let mut editor = Editor::<Remove>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.remove(inst_ref(entry, 0)), Err(Unremovable::Effect));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);

    let (name, int, writing) = (f.name("a"), f.int, f.call(f.pure_writing));
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    builder.push(writing, int, at(1));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let mut editor = Editor::<Remove>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.remove(inst_ref(entry, 0)), Err(Unremovable::Effect));
}

#[test]
fn remove_refuses_a_check_a_store_and_an_idempotent_call() {
    let mut f = fixture();
    let (name, int, idempotent) = (f.name("a"), f.int, f.call(f.idempotent));
    let mut builder = f.build(true, f.int);
    let a = builder.param(name, int);
    let x = builder.slot(int, at(0));
    builder.push_unit(check_add(a, a), at(1));
    builder.push_unit(Op::Assign { target: RefTarget::Var(x), path: Vec::new(), value: a, restores: false }, at(2));
    builder.push_effect(idempotent, int, at(3));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));

    let mut facts = NoFacts;
    let mut editor = Editor::<Remove>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.remove(inst_ref(entry, 0)), Err(Unremovable::Traps));
    assert_eq!(editor.remove(inst_ref(entry, 1)), Err(Unremovable::Writes));
    assert_eq!(editor.remove(inst_ref(entry, 2)), Err(Unremovable::Effect));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

#[test]
fn remove_takes_out_an_unused_field_set_which_writes_no_storage() {
    let mut f = fixture();
    let (na, np, nx, int, object) = (f.name("a"), f.name("p"), f.name("x"), f.int, f.object);
    let mut builder = f.build(false, int);
    let a = builder.param(na, int);
    let p = builder.param(np, object);
    builder.push(Op::FieldSet { object: p, field: nx, rest: Vec::new(), value: a }, object, at(1));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Remove>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.remove(inst_ref(entry, 0)), Ok(()));
    editor.finish();
    assert!(f.insts(id, entry).is_empty());
}

#[test]
fn removing_a_taking_removes_its_start() {
    let mut f = fixture();
    let (name, int, call) = (f.name("a"), f.int, f.call(f.pure_splittable));
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    builder.push(call, int, at(1));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);

    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Effect>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.split_call(inst_ref(entry, 0)), Ok(()));
    editor.finish();
    assert_eq!(f.insts(id, entry).len(), 2);

    let mut none = NoFacts;
    let mut editor = Editor::<Remove>::new(&mut f.module, id, &mut none);
    assert_eq!(editor.remove(inst_ref(entry, 1)), Ok(()));
    editor.finish();
    assert!(f.insts(id, entry).is_empty());
}

// -- retarget, take_arm, prune ---------------------------------------------

/// `if c { then } else { other }` joining at `join`, where `then` computes
/// `x = a + a` the join never reads, and an unused block `spare` that also
/// returns.
fn branching(f: &mut Fixture) -> (BodyId, [BlockId; 4]) {
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let [then_block, else_block, join, spare] = [(); 4].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join));
    builder.switch_to(then_block);
    builder.push(add(a, a), int, at(1));
    builder.terminate(jump(join));
    builder.switch_to(else_block);
    builder.terminate(jump(join));
    builder.switch_to(join);
    builder.terminate(ret(a));
    builder.switch_to(spare);
    builder.terminate(ret(a));
    (builder.seal().expect("the body has no open state"), [then_block, else_block, join, spare])
}

#[test]
fn retarget_moves_every_edge_and_the_diamonds_join() {
    let mut f = fixture();
    let (id, [_, else_block, join, spare]) = branching(&mut f);
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.retarget(join, spare);
    editor.finish();
    let Terminator::Diamond { join: now, .. } = f.body(id).block(entry).terminator() else {
        panic!("the entry still branches")
    };
    assert_eq!(*now, spare);
    assert!(matches!(f.body(id).block(else_block).terminator(), Terminator::Jump { to, .. } if *to == spare));
}

#[test]
fn take_arm_makes_a_diamond_a_jump_and_its_join_stays_a_block() {
    let mut f = fixture();
    let (id, [then_block, _, join, _]) = branching(&mut f);
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.take_arm(entry, false);
    editor.finish();
    let body = f.body(id);
    assert!(matches!(body.block(entry).terminator(), Terminator::Jump { .. }));
    assert!(body.block(join).params().is_empty());
    assert_eq!(body.users(result_of(&body.block(then_block).insts()[0])).count(), 0);
}

#[test]
fn take_arm_on_a_jump_if_passes_the_taken_arms_arguments() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let [left, right] = [(); 2].map(|()| builder.new_block(&[int]));
    builder.terminate(Terminator::JumpIf { cond: c, then_to: left, then_args: vec![a], else_to: right, else_args: vec![a] });
    for block in [left, right] {
        builder.switch_to(block);
        let param = builder.block_params(block)[0];
        builder.terminate(ret(param));
    }
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.take_arm(entry, true);
    editor.finish();
    assert!(matches!(f.body(id).block(entry).terminator(), Terminator::Jump { to, args } if *to == left && *args == [a]));
    assert_eq!(f.body(id).use_count(c), 0);
}

#[test]
fn prune_removes_unreached_blocks_and_renumbers_the_rest() {
    let mut f = fixture();
    let (id, _) = branching(&mut f);
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Sccp>::new(&mut f.module, id, &mut facts);
    editor.take_arm(entry, false);
    editor.prune();
    editor.finish();
    let body = f.body(id);
    assert_eq!(body.blocks().count(), 3, "the untaken arm and the spare block are gone");
    let Terminator::Jump { to: else_block, .. } = body.block(entry).terminator() else {
        panic!("the entry jumps into its taken arm")
    };
    assert!(matches!(body.block(*else_block).terminator(), Terminator::Jump { to, .. } if to.index() < 3));
    assert_eq!(facts.refreshed.len(), 2);
}

#[test]
fn prune_keeps_a_surviving_diamonds_join_that_no_edge_reaches() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let [then_block, else_block, join, spare] = [(); 4].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join));
    for block in [then_block, else_block, join, spare] {
        builder.switch_to(block);
        builder.terminate(ret(a));
    }
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.prune();
    editor.finish();
    let body = f.body(id);
    assert_eq!(body.blocks().count(), 4, "only the spare block goes");
    let Terminator::Diamond { join, .. } = body.block(entry).terminator() else {
        panic!("the entry still branches")
    };
    assert!(join.index() < 4);
}

// -- move_to ---------------------------------------------------------------

#[test]
fn move_to_moves_an_instruction_whose_operands_dominate_its_new_place() {
    let mut f = fixture();
    let (id, a) = { let callee = f.opaque; two_calls(&mut f, callee) };
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Motion>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.move_to(inst_ref(entry, 2), Position::Before(inst_ref(entry, 0))), Ok(()));
    editor.finish();
    assert!(matches!(f.insts(id, entry)[0].op(), Op::Check { left, .. } if *left == a));
}

#[test]
fn move_to_refuses_a_place_an_operand_does_not_dominate() {
    let mut f = fixture();
    let (name, int) = (f.name("a"), f.int);
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    let x = builder.push(add(a, a), int, at(1));
    builder.push_unit(check_add(x, a), at(2));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Motion>::new(&mut f.module, id, &mut facts);
    let refused = editor.move_to(inst_ref(entry, 1), Position::Before(inst_ref(entry, 0)));
    editor.finish();
    assert_eq!(refused, Err(CannotMove::OperandNotDominating(x)));
    assert_eq!(snapshot(f.body(id)), before);
}

#[test]
fn move_to_refuses_crossing_a_conflicting_write() {
    let mut f = fixture();
    let (id, _) = { let callee = f.opaque; two_calls(&mut f, callee) };
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    facts.conflicting.insert((inst_ref(entry, 1), inst_ref(entry, 2)));
    let mut editor = Editor::<Motion>::new(&mut f.module, id, &mut facts);
    let refused = editor.move_to(inst_ref(entry, 2), Position::Before(inst_ref(entry, 0)));
    editor.finish();
    assert_eq!(refused, Err(CannotMove::CrossesConflictingWrite(inst_ref(entry, 1))));
}

#[test]
fn moving_an_effect_across_another_on_its_chain_is_refused_by_dominance() {
    let mut f = fixture();
    let (id, _) = { let callee = f.opaque; two_calls(&mut f, callee) };
    let entry = entry_of(&f, id);
    let first_yield = edge_of(&f.insts(id, entry)[0]).after;
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Motion>::new(&mut f.module, id, &mut facts);
    let refused = editor.move_to(inst_ref(entry, 1), Position::Before(inst_ref(entry, 0)));
    editor.finish();
    assert_eq!(refused, Err(CannotMove::OperandNotDominating(first_yield)));
    assert_eq!(snapshot(f.body(id)), before);
}

#[test]
fn moving_an_effect_away_from_what_its_token_reaches_is_refused() {
    let mut f = fixture();
    let (nc, int, bool, call) = (f.name("c"), f.int, f.bool, f.call(f.opaque));
    let mut builder = f.build(true, bool);
    let c = builder.param(nc, bool);
    builder.push_effect(call, int, at(1));
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join));
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(jump(join));
    }
    builder.switch_to(join);
    builder.terminate(ret(c));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));

    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Motion>::new(&mut f.module, id, &mut facts);
    let refused = editor.move_to(inst_ref(entry, 0), Position::End(then_block));
    editor.finish();
    assert_eq!(refused, Err(CannotMove::UseNotDominated(Site::Term(join))));
    assert_eq!(snapshot(f.body(id)), before);
}

/// `while c { v = op(a, a); w = v + v }` with `a` a parameter; `op` is the
/// instruction under test, the body's first.
fn loop_computing(f: &mut Fixture, op: fn(ValueId, ValueId) -> Op) -> (BodyId, BlockId) {
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header));
    builder.switch_to(header);
    builder.terminate(Terminator::While { cond: c, stages: Stages::lowered(loop_body), exit, exit_args: Vec::new() });
    builder.switch_to(loop_body);
    let v = builder.push(op(a, a), int, at(1));
    builder.push(Op::BinOp { op: BinOp::Max, left: v, right: v }, int, at(2));
    builder.terminate(jump(header));
    builder.switch_to(exit);
    builder.terminate(ret(a));
    (builder.seal().expect("the body has no open state"), loop_body)
}

fn min(left: ValueId, right: ValueId) -> Op {
    Op::BinOp { op: BinOp::Min, left, right }
}

#[test]
fn hoisting_a_pure_value_that_never_traps_out_of_a_loop_succeeds() {
    let mut f = fixture();
    let (id, loop_body) = loop_computing(&mut f, min);
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Motion>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.move_to(inst_ref(loop_body, 0), Position::End(entry)), Ok(()));
    editor.finish();
    assert!(matches!(f.insts(id, entry), [hoisted] if matches!(hoisted.op(), Op::BinOp { op: BinOp::Min, .. })));
    assert_eq!(f.insts(id, loop_body).len(), 1);
}

#[test]
fn hoisting_a_trapping_value_out_of_a_loop_is_refused() {
    let mut f = fixture();
    let (id, loop_body) = loop_computing(&mut f, add);
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Motion>::new(&mut f.module, id, &mut facts);
    let refused = editor.move_to(inst_ref(loop_body, 0), Position::End(entry));
    editor.finish();
    assert_eq!(refused, Err(CannotMove::ChangesWhereItRuns));
    assert_eq!(snapshot(f.body(id)), before);
}

#[test]
fn a_split_calls_start_and_taking_move_while_their_tokens_still_dominate() {
    let mut f = fixture();
    let (name, int, call) = (f.name("a"), f.int, f.call(f.splittable));
    let mut builder = f.build(true, f.int);
    let a = builder.param(name, int);
    builder.push_effect(call, int, at(1));
    builder.push_unit(check_add(a, a), at(2));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);

    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Effect>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.split_call(inst_ref(entry, 0)), Ok(()));
    editor.finish();

    let mut editor = Editor::<Motion>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.move_to(inst_ref(entry, 1), Position::End(entry)), Ok(()));
    assert_eq!(editor.move_to(inst_ref(entry, 0), Position::Before(inst_ref(entry, 2))), Ok(()));
    editor.finish();
    let ops: Vec<&Op> = f.insts(id, entry).iter().map(Inst::op).collect();
    assert!(matches!(ops[..], [Op::Check { .. }, Op::Spawn { .. }, Op::TakeEffect { .. }]), "{ops:?}");
}

// -- split_call --------------------------------------------------------------

#[test]
fn split_call_makes_a_start_and_its_taking() {
    let mut f = fixture();
    let (id, _) = { let callee = f.splittable; two_calls(&mut f, callee) };
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Effect>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.split_call(inst_ref(entry, 1)), Ok(()));
    editor.finish();
    let insts = f.insts(id, entry);
    assert!(matches!(insts[1].op(), Op::Spawn { .. }));
    assert!(matches!(insts[2].op(), Op::TakeEffect { .. }));
    assert_eq!(facts.refreshed.len(), 1);
}

#[test]
fn split_call_refuses_a_callee_that_does_not_let_it_split() {
    let mut f = fixture();
    let (id, _) = { let callee = f.opaque; two_calls(&mut f, callee) };
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Effect>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.split_call(inst_ref(entry, 0)), Err(CannotSplit::NotSplittable));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

// -- promote -----------------------------------------------------------------

fn take(slot: ValueId) -> Op {
    Op::Take { target: RefTarget::Var(slot), path: Vec::new(), taken_out: false }
}

fn assign(slot: ValueId, value: ValueId) -> Op {
    Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }
}

/// `x = 1; if c { x = 2 }; while c { x = x + 1 }; return x`, with `x` a
/// slot.
fn slot_across_a_diamond_and_a_loop(f: &mut Fixture) -> (BodyId, ValueId, [BlockId; 6]) {
    let (nc, int, bool) = (f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let c = builder.param(nc, bool);
    let x = builder.slot(int, at(30));
    let one = builder.push(int_const(1), int, at(1));
    builder.push_unit(assign(x, one), at(2));
    let [then_block, else_block, join, loop_body, exit] = [(); 5].map(|()| builder.new_block(&[]));
    let header = builder.new_block(&[]);
    builder.terminate(diamond(c, then_block, else_block, join));
    builder.switch_to(then_block);
    let two = builder.push(int_const(2), int, at(3));
    builder.push_unit(assign(x, two), at(4));
    builder.terminate(jump(join));
    builder.switch_to(else_block);
    builder.terminate(jump(join));
    builder.switch_to(join);
    builder.terminate(jump(header));
    builder.switch_to(header);
    builder.terminate(Terminator::While { cond: c, stages: Stages::lowered(loop_body), exit, exit_args: Vec::new() });
    builder.switch_to(loop_body);
    let read = builder.push(take(x), int, at(5));
    let step = builder.push(int_const(1), int, at(6));
    let next = builder.push(add(read, step), int, at(7));
    builder.push_unit(assign(x, next), at(8));
    builder.terminate(jump(header));
    builder.switch_to(exit);
    let last = builder.push(take(x), int, at(9));
    builder.terminate(ret(last));
    let id = builder.seal().expect("the body has no open state");
    (id, x, [then_block, else_block, join, header, loop_body, exit])
}

#[test]
fn promote_builds_values_across_a_diamond_and_a_loop() {
    let mut f = fixture();
    let (id, _, [then_block, else_block, join, header, loop_body, exit]) = slot_across_a_diamond_and_a_loop(&mut f);
    let x = f.body(id).slots()[0];
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Storage>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.promote(x), Ok(()));
    editor.finish();

    let body = f.body(id);
    assert!(body.slots().is_empty());
    let storage_ops = body
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .filter(|inst| matches!(inst.op(), Op::Take { .. } | Op::Assign { .. } | Op::Undef))
        .count();
    assert_eq!(storage_ops, 0, "every Take and Assign of the slot is gone, and no Undef is needed");
    let one = result_of(&body.block(entry).insts()[0]);
    let two = result_of(&body.block(then_block).insts()[0]);
    let [joined] = body.block(join).params() else {
        panic!("the join takes the slot's value from its arms: {:?}", body.block(join).params())
    };
    let passed = |block: BlockId| match body.block(block).terminator() {
        Terminator::Jump { args, .. } => args.clone(),
        other => panic!("{other:?} is a jump"),
    };
    assert_eq!((passed(then_block), passed(else_block)), (vec![two], vec![one]));
    let [carried] = body.block(header).params() else {
        panic!("the loop header takes the slot's value around the loop: {:?}", body.block(header).params())
    };
    assert_eq!(passed(join), vec![*joined]);
    let body_insts = body.block(loop_body).insts();
    assert!(matches!(body_insts[1].op(), Op::BinOp { left, .. } if left == carried));
    assert_eq!(passed(loop_body), vec![result_of(&body_insts[1])]);
    assert!(matches!(body.block(exit).terminator(), Terminator::Return { value, .. } if value == carried));
}

#[test]
fn promote_reads_undef_where_no_store_comes_first() {
    let mut f = fixture();
    let int = f.int;
    let mut builder = f.build(false, f.int);
    let x = builder.slot(int, at(30));
    let read = builder.push(take(x), int, at(1));
    builder.terminate(ret(read));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Storage>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.promote(x), Ok(()));
    editor.finish();
    let insts = f.insts(id, entry);
    assert!(matches!(insts, [undef] if matches!(undef.op(), Op::Undef) && undef.span() == at(30)));
    assert!(matches!(f.body(id).block(entry).terminator(), Terminator::Return { value, .. } if *value == result_of(&insts[0])));
}

#[test]
fn promote_refuses_a_slot_whose_reference_escapes() {
    let mut f = fixture();
    let (id, x, [.., loop_body, _]) = slot_across_a_diamond_and_a_loop(&mut f);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    facts.escaping.insert(x, inst_ref(loop_body, 0));
    let mut editor = Editor::<Storage>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.promote(x), Err(CannotPromote::Escapes(inst_ref(loop_body, 0))));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

// -- cursor ----------------------------------------------------------------

#[test]
fn a_cursor_walks_on_across_the_removals_made_through_it() {
    let mut f = fixture();
    let (na, int) = (f.name("a"), f.int);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let kept = builder.push(min(a, a), int, at(1));
    builder.push(min(a, a), int, at(2));
    builder.push(min(a, a), int, at(3));
    let [next_block] = [(); 1].map(|()| builder.new_block(&[]));
    builder.terminate(jump(next_block));
    builder.switch_to(next_block);
    builder.push(min(a, a), int, at(4));
    builder.terminate(ret(kept));
    let id = builder.seal().expect("the body has no open state");

    let mut facts = NoFacts;
    let mut editor = Editor::<Remove>::new(&mut f.module, id, &mut facts);
    let mut cursor = editor.cursor();
    let mut seen = Vec::new();
    while let Some(inst) = cursor.next() {
        let span = cursor.editor().inst(inst).span();
        seen.push(span);
        let _ = cursor.editor().remove(inst);
    }
    editor.finish();
    assert_eq!(seen, vec![at(1), at(2), at(3), at(4)], "each instruction is yielded once");
    let entry = entry_of(&f, id);
    assert_eq!(f.insts(id, entry).len(), 1);
    assert!(f.insts(id, next_block).is_empty());
}

#[test]
fn a_cursor_skips_what_is_inserted_before_it_and_yields_what_is_inserted_after() {
    let mut f = fixture();
    let (id, [a, ..]) = straight_line(&mut f);
    let (entry, int) = (entry_of(&f, id), f.int);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    let mut cursor = editor.cursor();
    let mut seen = Vec::new();
    while let Some(inst) = cursor.next() {
        let span = cursor.editor().inst(inst).span();
        seen.push(span);
        if span == at(1) {
            cursor.editor().insert(Position::Before(inst), neg(a), int, at(20));
            let after = InstRef { block: inst.block, index: inst.index + 2 };
            cursor.editor().insert(Position::Before(after), neg(a), int, at(21));
        }
    }
    editor.finish();
    assert_eq!(seen, vec![at(1), at(21), at(2), at(3), at(4)]);
    let spans: Vec<Span> = f.insts(id, entry).iter().map(Inst::span).collect();
    assert_eq!(spans, vec![at(20), at(1), at(21), at(2), at(3), at(4)]);
}

// -- debug checks ------------------------------------------------------------

#[test]
#[should_panic(expected = "outside the scope's")]
fn a_change_outside_the_scope_is_caught() {
    let mut f = fixture();
    let (id, [_, _, join, spare]) = branching(&mut f);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    editor.edit_directly(|module| {
        let mut rewriter = module.rewrite(id);
        rewriter.retarget(join, spare);
        rewriter.finish()
    });
}

#[test]
#[should_panic(expected = "broke StructureKept")]
fn a_broken_invariant_is_caught() {
    let mut f = fixture();
    let (id, _) = branching(&mut f);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.edit_directly(|module| {
        let body = &mut module.bodies[id.0 as usize];
        let entry = body.entry;
        let Terminator::Diamond { join, .. } = &mut body.blocks[entry.index()].terminator else {
            panic!("the entry branches")
        };
        *join = BlockId::of_index(99);
        let mut changes = Changes::default();
        changes.record(Aspect::CfgShape);
        changes
    });
}

// -- Keeps checkers ------------------------------------------------------------

fn body_mut(f: &mut Fixture, id: BodyId) -> &mut Body {
    &mut f.module.bodies[id.0 as usize]
}

#[test]
fn types_agree_holds_on_a_built_body_and_catches_a_mistyped_result() {
    let mut f = fixture();
    let (id, [a, ..]) = straight_line(&mut f);
    assert_eq!(<TypesAgree as Invariant>::check(f.body(id), f.module.types()), Ok(()));
    let entry = entry_of(&f, id);
    body_mut(&mut f, id).blocks[entry.index()].insts[0].op = Op::BinOp { op: BinOp::Lt, left: a, right: a };
    let refused = <TypesAgree as Invariant>::check(f.body(id), f.module.types());
    assert!(matches!(refused, Err(Violation { invariant: "TypesAgree", .. })), "{refused:?}");
}

#[test]
fn def_dominates_uses_holds_on_a_built_body_and_catches_a_use_before_its_definition() {
    let mut f = fixture();
    let (id, [a, _, y, _]) = straight_line(&mut f);
    assert_eq!(<DefDominatesUses as Invariant>::check(f.body(id), f.module.types()), Ok(()));
    let entry = entry_of(&f, id);
    body_mut(&mut f, id).blocks[entry.index()].insts[0].op = add(a, y);
    let refused = <DefDominatesUses as Invariant>::check(f.body(id), f.module.types());
    assert!(matches!(refused, Err(Violation { invariant: "DefDominatesUses", .. })), "{refused:?}");
}

#[test]
fn order_chain_whole_holds_on_a_built_body_and_catches_a_dropped_token() {
    let mut f = fixture();
    let (id, _) = { let callee = f.opaque; two_calls(&mut f, callee) };
    assert_eq!(<OrderChainWhole as Invariant>::check(f.body(id), f.module.types()), Ok(()));
    let entry = entry_of(&f, id);
    let o0 = f.body(id).order_param().expect("the body is ordered");
    let Terminator::Return { order, .. } = &mut body_mut(&mut f, id).blocks[entry.index()].terminator else {
        panic!("the entry returns")
    };
    *order = Some(o0);
    let refused = <OrderChainWhole as Invariant>::check(f.body(id), f.module.types());
    assert!(matches!(refused, Err(Violation { invariant: "OrderChainWhole", .. })), "{refused:?}");
}

#[test]
fn structure_kept_holds_on_a_built_body_and_catches_a_join_that_is_no_block() {
    let mut f = fixture();
    let (id, _) = branching(&mut f);
    assert_eq!(<StructureKept as Invariant>::check(f.body(id), f.module.types()), Ok(()));
    let entry = entry_of(&f, id);
    let Terminator::Diamond { join, .. } = &mut body_mut(&mut f, id).blocks[entry.index()].terminator else {
        panic!("the entry branches")
    };
    *join = BlockId::of_index(99);
    let refused = <StructureKept as Invariant>::check(f.body(id), f.module.types());
    assert!(matches!(refused, Err(Violation { invariant: "StructureKept", .. })), "{refused:?}");
}

#[test]
fn split_taken_once_holds_after_a_split_and_catches_a_start_never_taken() {
    let mut f = fixture();
    let (id, _) = { let callee = f.splittable; two_calls(&mut f, callee) };
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Effect>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.split_call(inst_ref(entry, 0)), Ok(()));
    editor.finish();
    assert_eq!(<SplitTakenOnce as Invariant>::check(f.body(id), f.module.types()), Ok(()));

    let handle = result_of(&f.insts(id, entry)[0]);
    body_mut(&mut f, id).blocks[entry.index()].insts[1].op = Op::Drop { src: handle };
    let refused = <SplitTakenOnce as Invariant>::check(f.body(id), f.module.types());
    assert!(matches!(refused, Err(Violation { invariant: "SplitTakenOnce", .. })), "{refused:?}");
}

#[test]
fn order_chain_whole_accepts_an_anyorder_fork_that_merges_and_rejects_one_that_does_not() {
    let mut f = fixture();
    let (int, calls) = (f.int, [f.call(f.opaque), f.call(f.opaque)]);
    let [first, second] = calls;
    let mut builder = f.build(true, f.int);
    builder.open_anyorder();
    let r = builder.push_effect(first, int, at(1));
    builder.push_effect(second, int, at(2));
    builder.close_anyorder();
    builder.terminate(ret(r));
    let id = builder.seal().expect("the body has no open state");
    assert_eq!(<OrderChainWhole as Invariant>::check(f.body(id), f.module.types()), Ok(()));

    let (id, _) = { let callee = f.opaque; two_calls(&mut f, callee) };
    let entry = entry_of(&f, id);
    let o0 = f.body(id).order_param().expect("the body is ordered");
    let Some(OrderUse::Edge(edge)) = &mut body_mut(&mut f, id).blocks[entry.index()].insts[1].order else {
        panic!("the second call is on the chain")
    };
    edge.before = o0;
    let refused = <OrderChainWhole as Invariant>::check(f.body(id), f.module.types());
    assert!(
        matches!(&refused, Err(Violation { invariant: "OrderChainWhole", detail }) if detail.contains("no Merge joins")),
        "{refused:?}"
    );
}

fn types_agree(f: &Fixture, id: BodyId) -> Result<(), Violation> {
    <TypesAgree as Invariant>::check(f.body(id), f.module.types())
}

#[test]
fn types_agree_checks_every_return_against_the_bodys_result_type() {
    let mut f = fixture();
    let (id, _) = branching(&mut f);
    assert_eq!(types_agree(&f, id), Ok(()));

    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let [left, right] = [(); 2].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::JumpIf { cond: c, then_to: left, then_args: Vec::new(), else_to: right, else_args: Vec::new() });
    builder.switch_to(left);
    builder.terminate(ret(a));
    builder.switch_to(right);
    builder.terminate(ret(c));
    let id = builder.seal().expect("the body has no open state");
    assert!(matches!(types_agree(&f, id), Err(Violation { invariant: "TypesAgree", .. })));
}

fn switching_on(f: &mut Fixture, key: SwitchKey) -> BodyId {
    let (na, int) = (f.name("a"), f.int);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let [arm, other] = [(); 2].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::Switch { tag: a, arms: vec![(key, arm, Vec::new())], default: Some((other, Vec::new())) });
    for block in [arm, other] {
        builder.switch_to(block);
        builder.terminate(ret(a));
    }
    builder.seal().expect("the body has no open state")
}

#[test]
fn types_agree_checks_switch_keys_against_the_scrutinee() {
    let mut f = fixture();
    let good = switching_on(&mut f, SwitchKey::Int(3));
    assert_eq!(types_agree(&f, good), Ok(()));
    let bad = switching_on(&mut f, SwitchKey::Bool(true));
    assert!(matches!(types_agree(&f, bad), Err(Violation { invariant: "TypesAgree", .. })));
}

fn range_loop(f: &mut Fixture, element: Ty) -> BodyId {
    let (na, int) = (f.name("a"), f.int);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let header = builder.new_block(&[]);
    let loop_body = builder.new_block(&[element]);
    let exit = builder.new_block(&[]);
    builder.terminate(jump(header));
    builder.switch_to(header);
    builder.terminate(Terminator::For {
        source: ForSource::Range { at: a, hi: a },
        stages: Stages::lowered(loop_body),
        exit,
        exit_trip: crate::ir::ExitTrip::Absent,
        exit_args: Vec::new(),
    });
    builder.switch_to(loop_body);
    builder.terminate(jump(header));
    builder.switch_to(exit);
    builder.terminate(ret(a));
    builder.seal().expect("the body has no open state")
}

#[test]
fn types_agree_checks_a_loops_source_against_what_it_hands_its_body() {
    let mut f = fixture();
    let (int, bool) = (f.int, f.bool);
    let good = range_loop(&mut f, int);
    assert_eq!(types_agree(&f, good), Ok(()));
    let bad = range_loop(&mut f, bool);
    assert!(matches!(types_agree(&f, bad), Err(Violation { invariant: "TypesAgree", .. })));
}

fn fetching(f: &mut Fixture, ty: Ty) -> BodyId {
    let (na, int, page) = (f.name("a"), f.int, QualifiedRef::root(f.name("page")));
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    builder.push(Op::Fetch { context: page }, ty, at(1));
    builder.terminate(ret(a));
    builder.seal().expect("the body has no open state")
}

#[test]
fn types_agree_checks_that_a_context_value_holds_no_position() {
    let mut f = fixture();
    let (int, str_ref) = (f.int, f.str_ref);
    let good = fetching(&mut f, int);
    assert_eq!(types_agree(&f, good), Ok(()));
    let bad = fetching(&mut f, str_ref);
    assert!(matches!(types_agree(&f, bad), Err(Violation { invariant: "TypesAgree", .. })));
}

fn unwrapping(f: &mut Fixture, ty: Ty) -> BodyId {
    let (na, nr, int, result) = (f.name("a"), f.name("r"), f.int, f.result);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let r = builder.param(nr, result);
    builder.push(Op::UnwrapVariant { src: r }, ty, at(1));
    builder.terminate(ret(a));
    builder.seal().expect("the body has no open state")
}

#[test]
fn types_agree_checks_a_result_payload_against_its_variants() {
    let mut f = fixture();
    let (bool, str_ref) = (f.bool, f.str_ref);
    let good = unwrapping(&mut f, bool);
    assert_eq!(types_agree(&f, good), Ok(()));
    let bad = unwrapping(&mut f, str_ref);
    assert!(matches!(types_agree(&f, bad), Err(Violation { invariant: "TypesAgree", .. })));
}
