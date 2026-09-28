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
        module: Module::new(types),
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
    escaping: FxHashMap<ValueId, Site>,
    unobserved: FxHashMap<InstRef, Unobserved>,
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

    fn escapes(&self, slot: ValueId) -> Option<Site> {
        self.escaping.get(&slot).copied()
    }
}

impl DischargeFacts for TestFacts {
    fn discharged(&self, inst: InstRef) -> Option<Unobserved> {
        self.unobserved.get(&inst).copied()
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
    editor.retarget(join, spare).expect("the join and the spare block take no parameters");
    editor.finish();
    let Terminator::Diamond { join: now, .. } = f.body(id).block(entry).terminator() else {
        panic!("the entry still branches")
    };
    assert_eq!(*now, spare);
    assert!(matches!(f.body(id).block(else_block).terminator(), Terminator::Jump { to, .. } if *to == spare));
}

/// `retarget` refuses the entry, blocks whose parameters differ, and a
/// block whose parameter a later instruction reads, and leaves the body as
/// it was.
#[test]
fn retarget_refuses_the_entry_disagreeing_parameters_and_a_parameter_read_beyond_the_jump() {
    let mut f = fixture();
    let (int, bool) = (f.int, f.bool);
    let mut builder = f.build(false, int);
    let one = builder.push(int_const(1), int, at(0));
    let [passes, reads, target] = [(); 3].map(|()| builder.new_block(&[int]));
    let takes_bool = builder.new_block(&[bool]);
    builder.terminate(Terminator::Jump { to: passes, args: vec![one] });
    builder.switch_to(passes);
    let p = builder.block_params(passes)[0];
    builder.terminate(Terminator::Jump { to: reads, args: vec![p] });
    builder.switch_to(reads);
    let r = builder.block_params(reads)[0];
    let doubled = builder.push(add(r, r), int, at(1));
    builder.terminate(Terminator::Jump { to: target, args: vec![doubled] });
    builder.switch_to(target);
    let t = builder.block_params(target)[0];
    builder.terminate(ret(t));
    builder.switch_to(takes_bool);
    builder.terminate(ret(one));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));

    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.retarget(entry, passes), Err(CannotRetarget::Entry));
    assert_eq!(editor.retarget(passes, takes_bool), Err(CannotRetarget::ParamsDisagree));
    assert_eq!(editor.retarget(reads, target), Err(CannotRetarget::ParamsUsedBeyond(Site::Inst(reads, 0))));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
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
    facts.escaping.insert(x, Site::Inst(loop_body, 0));
    let mut editor = Editor::<Storage>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.promote(x), Err(CannotPromote::Escapes(Site::Inst(loop_body, 0))));
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
    let [arm, other, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::Switch { tag: a, arms: vec![(key, arm, Vec::new())], default: Some((other, Vec::new())), join });
    for block in [arm, other, join] {
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

// -- D57: a Switch holds its join --------------------------------------------

/// `switch a { 1 => one, 2 => two, _ => other }` joining at `join`, and an
/// unused block `spare`; the arms jump to the join where `arms_join`, and
/// return otherwise.
fn switching(f: &mut Fixture, arms_join: bool) -> (BodyId, ValueId, [BlockId; 5]) {
    let (na, int) = (f.name("a"), f.int);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let [one, two, other, join, spare] = [(); 5].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::Switch {
        tag: a,
        arms: vec![(SwitchKey::Int(1), one, Vec::new()), (SwitchKey::Int(2), two, Vec::new())],
        default: Some((other, Vec::new())),
        join,
    });
    for block in [one, two, other] {
        builder.switch_to(block);
        builder.terminate(if arms_join { jump(join) } else { ret(a) });
    }
    for block in [join, spare] {
        builder.switch_to(block);
        builder.terminate(ret(a));
    }
    (builder.seal().expect("the body has no open state"), a, [one, two, other, join, spare])
}

#[test]
fn take_case_takes_the_arm_of_a_known_key_and_the_join_stays_a_block() {
    let mut f = fixture();
    let (id, a, [_, two, _, join, _]) = switching(&mut f, true);
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.take_case(entry, &SwitchKey::Int(2));
    editor.finish();
    let body = f.body(id);
    assert!(matches!(body.block(entry).terminator(), Terminator::Jump { to, .. } if *to == two));
    assert!(matches!(body.block(join).terminator(), Terminator::Return { .. }));
    assert_eq!(body.use_count(a), 2, "the tag's use went with the Switch");
}

#[test]
fn take_case_takes_the_default_for_a_key_no_arm_has() {
    let mut f = fixture();
    let (id, _, [_, _, other, _, _]) = switching(&mut f, true);
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.take_case(entry, &SwitchKey::Int(9));
    editor.finish();
    assert!(matches!(f.body(id).block(entry).terminator(), Terminator::Jump { to, .. } if *to == other));
}

#[test]
fn retarget_moves_a_switchs_join() {
    let mut f = fixture();
    let (id, _, [one, _, _, join, spare]) = switching(&mut f, true);
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.retarget(join, spare).expect("the join and the spare block take no parameters");
    editor.finish();
    let body = f.body(id);
    assert!(matches!(body.block(entry).terminator(), Terminator::Switch { join, .. } if *join == spare));
    assert!(matches!(body.block(one).terminator(), Terminator::Jump { to, .. } if *to == spare));
}

#[test]
fn prune_keeps_a_surviving_switchs_join_that_no_edge_reaches() {
    let mut f = fixture();
    let (id, _, _) = switching(&mut f, false);
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.prune();
    editor.finish();
    let body = f.body(id);
    assert_eq!(body.blocks().count(), 5, "only the spare block goes");
    let Terminator::Switch { join, .. } = body.block(entry).terminator() else {
        panic!("the entry still switches")
    };
    assert!(join.index() < 5);
    assert!(matches!(body.block(*join).terminator(), Terminator::Return { .. }));
}

#[test]
fn structure_kept_catches_a_switch_join_that_is_no_block() {
    let mut f = fixture();
    let (id, _, _) = switching(&mut f, true);
    assert_eq!(<StructureKept as Invariant>::check(f.body(id), f.module.types()), Ok(()));
    let entry = entry_of(&f, id);
    let Terminator::Switch { join, .. } = &mut body_mut(&mut f, id).blocks[entry.index()].terminator else {
        panic!("the entry switches")
    };
    *join = BlockId::of_index(99);
    let refused = <StructureKept as Invariant>::check(f.body(id), f.module.types());
    assert!(matches!(refused, Err(Violation { invariant: "StructureKept", .. })), "{refused:?}");
}

// -- G2: split_edge and merge_into_predecessor --------------------------------

/// `entry(a, c): if c { left(a) } else { right(a) }`, each arm returning
/// its parameter, as a `JumpIf`.
fn forking(f: &mut Fixture) -> (BodyId, ValueId, [BlockId; 2]) {
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
    (builder.seal().expect("the body has no open state"), a, [left, right])
}

#[test]
fn split_edge_puts_a_block_on_the_edge_that_passes_its_arguments_on() {
    let mut f = fixture();
    let (id, a, [left, right]) = forking(&mut f);
    let (entry, int) = (entry_of(&f, id), f.int);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    let middle = editor.split_edge(entry, left);
    let changes = editor.finish();

    assert!(changes.touched(Aspect::CfgShape));
    let body = f.body(id);
    let Terminator::JumpIf { then_to, then_args, else_to, .. } = body.block(entry).terminator() else {
        panic!("the entry still branches")
    };
    assert_eq!((*then_to, then_args.as_slice(), *else_to), (middle, [a].as_slice(), right));
    let [param] = body.block(middle).params() else {
        panic!("the new block takes what the edge passed")
    };
    assert_eq!(body.ty_of(*param), int);
    assert!(body.block(middle).insts().is_empty());
    assert!(matches!(body.block(middle).terminator(), Terminator::Jump { to, args } if *to == left && *args == [*param]));
}

#[test]
fn split_edge_into_a_loops_body_makes_the_new_block_the_loops_body() {
    let mut f = fixture();
    let (id, loop_body) = loop_computing(&mut f, min);
    let header = BlockId::of_index(1);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    let middle = editor.split_edge(header, loop_body);
    editor.finish();
    let body = f.body(id);
    let Terminator::While { stages, .. } = body.block(header).terminator() else {
        panic!("the header still loops")
    };
    assert_eq!(stages.body(), middle);
    assert!(matches!(body.block(middle).terminator(), Terminator::Jump { to, .. } if *to == loop_body));
}

#[test]
#[should_panic(expected = "split_edge names exactly one")]
fn split_edge_of_two_edges_to_one_block_is_a_callers_defect() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let both = builder.new_block(&[]);
    builder.terminate(Terminator::JumpIf { cond: c, then_to: both, then_args: Vec::new(), else_to: both, else_args: Vec::new() });
    builder.switch_to(both);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    editor.split_edge(entry, both);
}

#[test]
fn merge_into_predecessor_joins_a_block_its_sole_predecessor_jumps_to() {
    let mut f = fixture();
    let (na, int) = (f.name("a"), f.int);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let x = builder.push(add(a, a), int, at(1));
    let next = builder.new_block(&[int]);
    builder.terminate(Terminator::Jump { to: next, args: vec![x] });
    builder.switch_to(next);
    let p = builder.block_params(next)[0];
    let y = builder.push(neg(p), int, at(2));
    builder.terminate(ret(y));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);

    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.merge_into_predecessor(next), Ok(()));
    editor.finish();
    let body = f.body(id);
    assert_eq!(body.blocks().count(), 1);
    let insts = body.block(entry).insts();
    assert_eq!(insts.len(), 2);
    assert!(matches!(insts[1].op(), Op::UnaryOp { operand, .. } if *operand == x), "the parameter became its argument");
    assert_eq!(result_of(&insts[1]), y);
    assert!(matches!(body.block(entry).terminator(), Terminator::Return { value, .. } if *value == y));
    assert_eq!(body.use_count(x), 1);
}

#[test]
fn merge_into_predecessor_refuses_a_join_and_a_loops_exit() {
    let mut f = fixture();
    let (id, [then_block, _, join, _]) = branching(&mut f);
    let before = snapshot(f.body(id));
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.merge_into_predecessor(join), Err(CannotMerge::Structural));
    assert_eq!(editor.merge_into_predecessor(then_block), Err(CannotMerge::NotStraight), "its predecessor branches");
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);

    let (id, _, [one, _, _, join, _]) = switching(&mut f, true);
    let before = snapshot(f.body(id));
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.merge_into_predecessor(join), Err(CannotMerge::Structural));
    assert_eq!(editor.merge_into_predecessor(one), Err(CannotMerge::NotStraight));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);

    let (id, loop_body) = loop_computing(&mut f, min);
    let exit = BlockId::of_index(3);
    let before = snapshot(f.body(id));
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.merge_into_predecessor(exit), Err(CannotMerge::Structural));
    assert_eq!(editor.merge_into_predecessor(loop_body), Err(CannotMerge::Structural));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

#[test]
fn merge_into_predecessor_refuses_a_block_two_blocks_enter() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let [left, right, meet] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::JumpIf { cond: c, then_to: left, then_args: Vec::new(), else_to: right, else_args: Vec::new() });
    for block in [left, right] {
        builder.switch_to(block);
        builder.terminate(jump(meet));
    }
    builder.switch_to(meet);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.merge_into_predecessor(meet), Err(CannotMerge::NotStraight));
    assert_eq!(editor.merge_into_predecessor(entry), Err(CannotMerge::NotStraight), "the entry has no predecessor");
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

// -- G3: replace_use -----------------------------------------------------------

#[test]
fn replace_use_changes_one_site_only() {
    let mut f = fixture();
    let (id, [_, x, _, z]) = straight_line(&mut f);
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.replace_use(Site::Inst(entry, 3), x, z), Ok(()));
    editor.finish();
    let body = f.body(id);
    assert_eq!((body.use_count(x), body.use_count(z)), (1, 2));
    assert!(matches!(body.block(entry).insts()[1].op(), Op::UnaryOp { operand, .. } if *operand == x));
    assert!(matches!(body.block(entry).insts()[3].op(), Op::BinOp { left, right, .. } if (*left, *right) == (z, z)));
}

#[test]
fn replace_use_refuses_a_value_that_does_not_dominate_the_site() {
    let mut f = fixture();
    let (id, [_, x, _, z]) = straight_line(&mut f);
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Value>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.replace_use(Site::Inst(entry, 1), x, z), Err(DoesNotDominate { use_at: Site::Inst(entry, 1) }));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

// -- G4: swap ------------------------------------------------------------------

/// An ordered body of one block: `r1 = f(); r2 = g(); r3 = h(); return a`.
fn three_calls(f: &mut Fixture) -> BodyId {
    let (name, int) = (f.name("a"), f.int);
    let calls = [f.call(f.opaque), f.call(f.opaque), f.call(f.opaque)];
    let mut builder = f.build(true, f.int);
    let a = builder.param(name, int);
    for (n, call) in calls.into_iter().enumerate() {
        builder.push_effect(call, int, at(n as u32 + 1));
    }
    builder.terminate(ret(a));
    builder.seal().expect("the body has no open state")
}

#[test]
fn swap_trades_two_commuting_calls_and_the_chain_stays_whole() {
    let mut f = fixture();
    let id = three_calls(&mut f);
    let entry = entry_of(&f, id);
    let o0 = f.body(id).order_param().expect("the body is ordered");
    let [first, second, third] = [0, 1, 2].map(|index| edge_of(&f.insts(id, entry)[index]));
    let [r1, r2] = [0, 1].map(|index| result_of(&f.insts(id, entry)[index]));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Effect>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.swap(inst_ref(entry, 0), inst_ref(entry, 1)), Ok(()));
    editor.finish();

    let insts = f.insts(id, entry);
    assert_eq!([result_of(&insts[0]), result_of(&insts[1])], [r2, r1]);
    let (now_first, now_second, now_third) = (edge_of(&insts[0]), edge_of(&insts[1]), edge_of(&insts[2]));
    assert_eq!((now_first.before, now_first.after), (o0, second.after), "g consumes what f consumed");
    assert_eq!((now_second.before, now_second.after), (second.after, first.after), "f consumes what g yields");
    assert_eq!(now_third.before, first.after, "what used g's yield uses f's");
    assert_eq!(now_third.after, third.after);
    assert_eq!(returned_order(f.body(id), entry), Some(third.after));
    assert_eq!(<OrderChainWhole as Invariant>::check(f.body(id), f.module.types()), Ok(()));
    assert_eq!(<DefDominatesUses as Invariant>::check(f.body(id), f.module.types()), Ok(()));
}

#[test]
fn swap_refuses_calls_not_adjacent_on_the_chain_and_calls_kept_in_order() {
    let mut f = fixture();
    let id = three_calls(&mut f);
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    facts.kept_in_order.insert((inst_ref(entry, 1), inst_ref(entry, 2)));
    let mut editor = Editor::<Effect>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.swap(inst_ref(entry, 0), inst_ref(entry, 2)), Err(CannotSwap::NotAdjacent));
    assert_eq!(editor.swap(inst_ref(entry, 1), inst_ref(entry, 0)), Err(CannotSwap::NotAdjacent));
    assert_eq!(editor.swap(inst_ref(entry, 1), inst_ref(entry, 2)), Err(CannotSwap::Ordered));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

#[test]
fn swap_refuses_a_call_that_reads_the_others_result() {
    let mut f = fixture();
    let (name, int) = (f.name("a"), f.int);
    let taking_int = f.module.types.intern(TyKind::Fn {
        params: Box::new([int]),
        ret: int,
        effect: EffectTerm::Known(CallEffect::opaque()),
        flows: FlowTerm::Known(Flows::none()),
        calls: Calls::Many,
        holds: Holds::Nothing,
    });
    let first = f.call(f.opaque);
    let callee = Callee::Direct(QualifiedRef::root(f.name("k")));
    let mut builder = f.build(true, f.int);
    let a = builder.param(name, int);
    let r = builder.push_effect(first, int, at(1));
    builder.push_effect(Op::FunctionCall { callee, callee_ty: taking_int, args: vec![r] }, int, at(2));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Effect>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.swap(inst_ref(entry, 0), inst_ref(entry, 1)), Err(CannotSwap::DependsOn(r)));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

// -- G1: discharge -------------------------------------------------------------

/// `check(a + a); s = a; v = fetch page; commit page = a; return a`, with
/// `s` a slot.
fn dischargeable(f: &mut Fixture) -> BodyId {
    let (name, int, page) = (f.name("a"), f.int, QualifiedRef::root(f.name("page")));
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    let s = builder.slot(int, at(0));
    builder.push_unit(check_add(a, a), at(1));
    builder.push_unit(assign(s, a), at(2));
    builder.push(Op::Fetch { context: page }, int, at(3));
    builder.push_unit(Op::Commit { context: page, value: a, wrote: true }, at(4));
    builder.terminate(ret(a));
    builder.seal().expect("the body has no open state")
}

#[test]
fn discharge_removes_a_check_a_dead_store_an_unread_load_and_an_unwritten_commit() {
    let mut f = fixture();
    let id = dischargeable(&mut f);
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let facts_of = [Unobserved::NeverFires, Unobserved::NeverRead, Unobserved::NeverRead, Unobserved::NeverWritten];
    for (index, fact) in facts_of.into_iter().enumerate() {
        facts.unobserved.insert(inst_ref(entry, index), fact);
    }
    let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
    for index in (0..4).rev() {
        assert_eq!(editor.discharge(inst_ref(entry, index)), Ok(()), "instruction {index}");
    }
    let changes = editor.finish();
    assert!(f.insts(id, entry).is_empty());
    assert!(changes.touched(Aspect::Storage));
}

#[test]
fn discharge_refuses_without_a_fact_and_with_the_wrong_fact() {
    let mut f = fixture();
    let id = dischargeable(&mut f);
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    facts.unobserved.insert(inst_ref(entry, 0), Unobserved::NeverRead);
    facts.unobserved.insert(inst_ref(entry, 1), Unobserved::NeverWritten);
    facts.unobserved.insert(inst_ref(entry, 3), Unobserved::NeverRead);
    let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.discharge(inst_ref(entry, 0)), Err(CannotDischarge::WrongFact), "a check needs NeverFires");
    assert_eq!(editor.discharge(inst_ref(entry, 1)), Err(CannotDischarge::WrongFact), "a store needs NeverRead");
    assert_eq!(editor.discharge(inst_ref(entry, 2)), Err(CannotDischarge::NotProven));
    assert_eq!(editor.discharge(inst_ref(entry, 3)), Err(CannotDischarge::WrongFact), "a commit needs NeverWritten");
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
    assert!(facts.refreshed.is_empty());
}

#[test]
#[should_panic(expected = "discharge removes a check, a store, a load or a commit")]
fn discharging_another_kind_is_a_callers_defect() {
    let mut f = fixture();
    let (id, _) = straight_line(&mut f);
    let entry = entry_of(&f, id);
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
    let _ = editor.discharge(inst_ref(entry, 0));
}

// -- S1: split_slot ------------------------------------------------------------

#[test]
fn split_slot_gives_each_field_of_a_tuple_its_own_slot() {
    let mut f = fixture();
    let (name, int) = (f.name("a"), f.int);
    let pair = f.module.types.intern(TyKind::Tuple(Box::new([int, int])));
    let int_ref = f.module.types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    let s = builder.slot(pair, at(0));
    let field = |index: usize| vec![crate::ir::PathSeg::Index(index)];
    for index in 0..2 {
        builder.push_unit(Op::Assign { target: RefTarget::Var(s), path: field(index), value: a, restores: false }, at(1));
    }
    let t = builder.push(Op::Take { target: RefTarget::Var(s), path: field(1), taken_out: false }, int, at(2));
    builder.push(Op::Ref { target: RefTarget::Var(s), path: field(0), mutability: Mutability::Shared }, int_ref, at(3));
    builder.terminate(ret(t));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);

    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Storage>::new(&mut f.module, id, &mut facts);
    let parts = editor.split_slot(s).expect("every use names a field");
    editor.finish();
    let [first, second] = parts[..] else {
        panic!("a pair splits into two slots")
    };
    let body = f.body(id);
    assert_eq!(body.slots(), [first, second]);
    assert_eq!((body.ty_of(first), body.ty_of(second)), (int, int));
    let place = |index: usize| body.block(entry).insts()[index].op().place().map(|(target, path)| (*target, path.to_vec()));
    assert_eq!(place(0), Some((RefTarget::Var(first), Vec::new())));
    assert_eq!(place(1), Some((RefTarget::Var(second), Vec::new())));
    assert_eq!(place(2), Some((RefTarget::Var(second), Vec::new())));
    assert_eq!(place(3), Some((RefTarget::Var(first), Vec::new())));
    assert_eq!((body.use_count(first), body.use_count(second)), (2, 2));
}

#[test]
fn split_slot_refuses_a_type_that_is_not_structural_and_a_use_of_the_whole() {
    let mut f = fixture();
    let (name, int, object) = (f.name("a"), f.int, f.object);
    let pair = f.module.types.intern(TyKind::Tuple(Box::new([int, int])));
    let mut builder = f.build(false, f.int);
    let a = builder.param(name, int);
    let scalar = builder.slot(int, at(0));
    let declared = builder.slot(object, at(0));
    let whole = builder.slot(pair, at(0));
    builder.push_unit(
        Op::Assign { target: RefTarget::Var(whole), path: vec![crate::ir::PathSeg::Index(0)], value: a, restores: false },
        at(1),
    );
    builder.push(take(whole), pair, at(2));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));

    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Storage>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.split_slot(scalar), Err(CannotSplitSlot::NotStructural));
    assert_eq!(editor.split_slot(declared), Err(CannotSplitSlot::NotStructural), "an extern-defined type is never split");
    assert_eq!(editor.split_slot(whole), Err(CannotSplitSlot::Escapes(Site::Inst(entry, 1))));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

// -- S2: inline ----------------------------------------------------------------

fn closure_ty(f: &mut Fixture, params: &[Ty], effect: CallEffect) -> Ty {
    let int = f.int;
    f.module.types.intern(TyKind::Fn {
        params: params.into(),
        ret: int,
        effect: EffectTerm::Known(effect),
        flows: FlowTerm::Known(Flows::none()),
        calls: Calls::Many,
        holds: Holds::Nothing,
    })
}

/// The callee `|x| { g(); x + k }`, capturing `k`, and the caller
/// `k = 1; f = the closure; r = f(a); h(); return r + r`, both ordered.
#[test]
fn inline_puts_a_straight_callee_in_the_calls_place_on_the_callers_chain() {
    let mut f = fixture();
    let (na, nx, nk, int) = (f.name("a"), f.name("x"), f.name("k"), f.int);
    let lambda_ty = closure_ty(&mut f, &[int], CallEffect::opaque());
    let (inner_call, after_call) = (f.call(f.opaque), f.call(f.opaque));
    let mut builder = f.build(true, int);
    let x = builder.param(nx, int);
    let k = builder.capture(nk, int);
    builder.push_effect(inner_call, int, at(1));
    let sum = builder.push(add(x, k), int, at(2));
    builder.terminate(ret(sum));
    let callee = builder.seal().expect("the callee has no open state");

    let mut builder = f.build(true, int);
    let a = builder.param(na, int);
    let one = builder.push(int_const(1), int, at(3));
    let closure = builder.push(Op::MakeClosure { body: callee, captures: vec![one] }, lambda_ty, at(4));
    let call = Op::FunctionCall { callee: Callee::Indirect(closure), callee_ty: lambda_ty, args: vec![a] };
    let r = builder.push_effect(call, int, at(5));
    builder.push_effect(after_call, int, at(6));
    let doubled = builder.push(add(r, r), int, at(7));
    builder.terminate(ret(doubled));
    let caller = builder.seal().expect("the caller has no open state");
    let entry = entry_of(&f, caller);
    let o0 = f.body(caller).order_param().expect("the caller is ordered");

    let mut facts = NoFacts;
    let mut editor = Editor::<Inline>::new(&mut f.module, caller, &mut facts);
    assert_eq!(editor.inline(inst_ref(entry, 2)), Ok(()));
    editor.finish();

    let body = f.body(caller);
    assert_eq!(body.block(entry).insts().len(), 2, "the call is gone from its block");
    let Terminator::Jump { to: copy, args } = body.block(entry).terminator() else {
        panic!("the caller's block enters the copy")
    };
    assert!(args.is_empty());
    let copied = body.block(*copy).insts();
    assert_eq!(edge_of(&copied[0]).before, o0, "the callee's chain continues the caller's");
    assert!(matches!(copied[1].op(), Op::BinOp { left, right, .. } if (*left, *right) == (a, one)));
    let Terminator::Jump { to: continuation, args } = body.block(*copy).terminator() else {
        panic!("the copy's return jumps to the continuation")
    };
    assert_eq!(args.as_slice(), [result_of(&copied[1]), edge_of(&copied[0]).after]);
    let [result, token] = body.block(*continuation).params() else {
        panic!("the continuation takes the result and the token")
    };
    let rest = body.block(*continuation).insts();
    assert_eq!(edge_of(&rest[0]).before, *token);
    assert!(matches!(rest[1].op(), Op::BinOp { left, right, .. } if (*left, *right) == (*result, *result)));
    for invariant in [
        <TypesAgree as Invariant>::check(body, f.module.types()),
        <DefDominatesUses as Invariant>::check(body, f.module.types()),
        <OrderChainWhole as Invariant>::check(body, f.module.types()),
    ] {
        assert_eq!(invariant, Ok(()));
    }
}

/// The callee `|x, c| if c { return x } else { return x + x }` and the
/// caller `f = the closure; r = f(a, c); return r + a`.
#[test]
fn inline_joins_every_return_of_the_callee_at_the_continuation() {
    let mut f = fixture();
    let (na, nc, nx, int, bool) = (f.name("a"), f.name("c"), f.name("x"), f.int, f.bool);
    let lambda_ty = closure_ty(&mut f, &[int, bool], CallEffect::pure());
    let mut builder = f.build(false, int);
    let x = builder.param(nx, int);
    let c = builder.param(nc, bool);
    let [left, right] = [(); 2].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::JumpIf { cond: c, then_to: left, then_args: Vec::new(), else_to: right, else_args: Vec::new() });
    builder.switch_to(left);
    builder.terminate(ret(x));
    builder.switch_to(right);
    let doubled = builder.push(add(x, x), int, at(1));
    builder.terminate(ret(doubled));
    let callee = builder.seal().expect("the callee has no open state");

    let mut builder = f.build(false, int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let closure = builder.push(Op::MakeClosure { body: callee, captures: Vec::new() }, lambda_ty, at(2));
    let call = Op::FunctionCall { callee: Callee::Indirect(closure), callee_ty: lambda_ty, args: vec![a, c] };
    let r = builder.push(call, int, at(3));
    let sum = builder.push(add(r, a), int, at(4));
    builder.terminate(ret(sum));
    let caller = builder.seal().expect("the caller has no open state");
    let entry = entry_of(&f, caller);

    let mut facts = NoFacts;
    let mut editor = Editor::<Inline>::new(&mut f.module, caller, &mut facts);
    assert_eq!(editor.inline(inst_ref(entry, 1)), Ok(()));
    editor.finish();

    let body = f.body(caller);
    let continuation = BlockId::of_index(1);
    let [result] = body.block(continuation).params() else {
        panic!("the continuation takes the result alone, as the call is on no chain")
    };
    assert!(matches!(body.block(continuation).insts()[0].op(), Op::BinOp { left, .. } if left == result));
    let preds = body.predecessors();
    assert_eq!(preds[&continuation].len(), 2, "both returns jump to the continuation");
    let passed: Vec<ValueId> = preds[&continuation]
        .iter()
        .map(|pred| match body.block(*pred).terminator() {
            Terminator::Jump { args, .. } => args[0],
            other => panic!("{pred:?} ends in {other:?}"),
        })
        .collect();
    assert!(passed.contains(&a), "the left return passes the argument the parameter stood for");
    assert_eq!(<TypesAgree as Invariant>::check(body, f.module.types()), Ok(()));
    assert_eq!(<DefDominatesUses as Invariant>::check(body, f.module.types()), Ok(()));
}

#[test]
fn inline_refuses_a_callee_without_a_body_and_the_callers_own_body() {
    let mut f = fixture();
    let (na, int) = (f.name("a"), f.int);
    let lambda_ty = closure_ty(&mut f, &[], CallEffect::pure());
    let direct = f.call(f.pure);
    let own = BodyId(u32::try_from(f.module.bodies.len()).expect("few bodies"));
    let mut builder = f.build(false, int);
    let a = builder.param(na, int);
    builder.push(direct, int, at(1));
    let closure = builder.push(Op::MakeClosure { body: own, captures: Vec::new() }, lambda_ty, at(2));
    let call = Op::FunctionCall { callee: Callee::Indirect(closure), callee_ty: lambda_ty, args: Vec::new() };
    builder.push(call, int, at(3));
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    assert_eq!(id, own);
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));

    let mut facts = NoFacts;
    let mut editor = Editor::<Inline>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.inline(inst_ref(entry, 0)), Err(CannotInline::NoBody));
    assert_eq!(editor.inline(inst_ref(entry, 2)), Err(CannotInline::Recursive));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);
}

// -- S3: insert_drop -----------------------------------------------------------

#[test]
fn insert_drop_stands_after_the_last_use_and_is_refused_before_a_use() {
    let mut f = fixture();
    let (id, [_, x, _, _]) = straight_line(&mut f);
    let entry = entry_of(&f, id);
    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Drops>::new(&mut f.module, id, &mut facts);
    assert_eq!(
        editor.insert_drop(x, Position::Before(inst_ref(entry, 2)), at(9)),
        Err(CannotDrop::UsedAfter(Site::Inst(entry, 3)))
    );
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);

    let mut editor = Editor::<Drops>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.insert_drop(x, Position::End(entry), at(9)), Ok(()));
    editor.finish();
    let insts = f.insts(id, entry);
    assert!(matches!(insts[4].op(), Op::Drop { src } if *src == x));
    assert_eq!(insts[4].span(), at(9));
}

#[test]
fn insert_drop_is_refused_inside_a_loop_before_a_use_the_back_edge_reaches() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.build(false, f.int);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let x = builder.push(add(a, a), int, at(1));
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header));
    builder.switch_to(header);
    builder.terminate(Terminator::While { cond: c, stages: Stages::lowered(loop_body), exit, exit_args: Vec::new() });
    builder.switch_to(loop_body);
    builder.push(neg(x), int, at(2));
    builder.push(neg(a), int, at(3));
    builder.terminate(jump(header));
    builder.switch_to(exit);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");

    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Drops>::new(&mut f.module, id, &mut facts);
    assert_eq!(
        editor.insert_drop(x, Position::Before(inst_ref(loop_body, 1)), at(9)),
        Err(CannotDrop::UsedAfter(Site::Inst(loop_body, 0)))
    );
    assert_eq!(editor.insert_drop(x, Position::End(exit), at(9)), Ok(()));
    editor.finish();
}

// -- The cursor across an inline and a merge -----------------------------------

/// What a cursor yields over the whole body, by instruction identity, with
/// `edit` run at each instruction it yields.
fn walk_with<S>(editor: &mut Editor<'_, S>, mut edit: impl FnMut(&mut Editor<'_, S>, InstRef)) -> Vec<super::super::body::InstId>
where
    S: Scope,
{
    let mut yielded = Vec::new();
    let mut cursor = editor.cursor();
    while let Some(at) = cursor.next() {
        yielded.push(cursor.editor().inst(at).id());
        edit(cursor.editor(), at);
    }
    yielded
}

/// Every instruction of the body was yielded, and none twice.
fn assert_yielded_once(body: &Body, yielded: &[super::super::body::InstId]) {
    let distinct: FxHashSet<_> = yielded.iter().copied().collect();
    assert_eq!(distinct.len(), yielded.len(), "no instruction is yielded twice: {yielded:?}");
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            assert!(distinct.contains(&inst.id()), "{inst:?} is walked");
        }
    }
}

/// The callee `|x| x + k`, capturing `k`; the caller
/// `one = 1; f = the closure; r = f(a); s = r + r; return s`. The cursor
/// inlines the call when it meets it.
#[test]
fn a_cursor_yields_each_instruction_once_through_an_inline() {
    let mut f = fixture();
    let (na, nx, nk, int) = (f.name("a"), f.name("x"), f.name("k"), f.int);
    let lambda_ty = closure_ty(&mut f, &[int], CallEffect::pure());
    let mut builder = f.build(false, int);
    let x = builder.param(nx, int);
    let k = builder.capture(nk, int);
    let sum = builder.push(add(x, k), int, at(1));
    builder.terminate(ret(sum));
    let callee = builder.seal().expect("the callee has no open state");

    let mut builder = f.build(false, int);
    let a = builder.param(na, int);
    let one = builder.push(int_const(1), int, at(2));
    let closure = builder.push(Op::MakeClosure { body: callee, captures: vec![one] }, lambda_ty, at(3));
    let r = builder.push(Op::FunctionCall { callee: Callee::Indirect(closure), callee_ty: lambda_ty, args: vec![a] }, int, at(4));
    let s = builder.push(add(r, r), int, at(5));
    builder.terminate(ret(s));
    let caller = builder.seal().expect("the caller has no open state");

    let mut facts = NoFacts;
    let mut editor = Editor::<Inline>::new(&mut f.module, caller, &mut facts);
    let yielded = walk_with(&mut editor, |editor, at| {
        if matches!(editor.inst(at).op(), Op::FunctionCall { .. }) {
            assert_eq!(editor.inline(at), Ok(()));
        }
    });
    editor.finish();
    assert_yielded_once(f.body(caller), &yielded);
    assert_eq!(yielded.len(), 5, "one, the closure, the call, s and the callee's sum");
}

/// `entry: x = a + a; jump pred`, `tail: t = -x; return t`,
/// `pred: z = a + a; jump tail`: the cursor yields `t` in `tail` before it
/// reaches `pred`, and there merges `tail` into `pred`, which moves `t`
/// ahead of the cursor again.
#[test]
fn a_cursor_yields_each_instruction_once_through_a_merge() {
    let mut f = fixture();
    let (na, int) = (f.name("a"), f.int);
    let mut builder = f.build(false, int);
    let a = builder.param(na, int);
    let x = builder.push(add(a, a), int, at(1));
    let [tail, pred] = [(); 2].map(|()| builder.new_block(&[]));
    builder.terminate(jump(pred));
    builder.switch_to(tail);
    let t = builder.push(neg(x), int, at(2));
    builder.terminate(ret(t));
    builder.switch_to(pred);
    builder.push(add(a, a), int, at(3));
    builder.terminate(jump(tail));
    let id = builder.seal().expect("the body has no open state");

    let mut facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(&mut f.module, id, &mut facts);
    let yielded = walk_with(&mut editor, |editor, at| {
        if editor.inst(at).result() == Some(t) {
            assert_eq!(editor.merge_into_predecessor(at.block), Ok(()));
        }
    });
    editor.finish();
    let body = f.body(id);
    assert_eq!(body.blocks().count(), 2);
    assert_yielded_once(body, &yielded);
    assert_eq!(yielded.len(), 3);
}

// -- G1 on a checked bound -----------------------------------------------------

#[test]
fn discharge_of_a_checked_index_proves_its_bound_and_keeps_the_index() {
    let mut f = fixture();
    let (ns, ni, int) = (f.name("s"), f.name("i"), f.int);
    let u64_ty = f.module.types.intern(TyKind::Int(IntTy::U64));
    let slice = f.module.types.intern(TyKind::Slice(int));
    let slice_ref = f.module.types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(slice)));
    let mut builder = f.build(false, int);
    let s = builder.param(ns, slice_ref);
    let i = builder.param(ni, u64_ty);
    let checked = Op::Index { slice: s, index: i, mode: crate::ir::IndexMode::Copy, bound: crate::ir::IndexBound::Checked };
    let e = builder.push(checked, int, at(1));
    builder.terminate(ret(e));
    let id = builder.seal().expect("the body has no open state");
    let entry = entry_of(&f, id);

    let before = snapshot(f.body(id));
    let mut facts = TestFacts::of(&f.module, id);
    let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.discharge(inst_ref(entry, 0)), Err(CannotDischarge::NotProven));
    editor.finish();
    assert_eq!(snapshot(f.body(id)), before);

    facts.unobserved.insert(inst_ref(entry, 0), Unobserved::NeverRead);
    let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.discharge(inst_ref(entry, 0)), Err(CannotDischarge::WrongFact), "a checked bound needs NeverFires");
    editor.finish();

    facts.unobserved.insert(inst_ref(entry, 0), Unobserved::NeverFires);
    let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
    assert_eq!(editor.discharge(inst_ref(entry, 0)), Ok(()));
    editor.finish();
    let insts = f.insts(id, entry);
    assert_eq!(insts.len(), 1, "the index stays");
    assert!(matches!(insts[0].op(), Op::Index { bound: crate::ir::IndexBound::Proven, .. }));
    assert_eq!(result_of(&insts[0]), e);
}
