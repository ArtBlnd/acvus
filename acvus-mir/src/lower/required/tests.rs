//! The required steps on bodies built with the Builder, each run through
//! its public entry over the whole module.

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, QualifiedRef};

use super::*;
use crate::decl::{ContextDecl, Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::cfg::Builder;
use crate::ty::{Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, TypeArg};

struct Fixture {
    module: Module,
    interner: Interner,
    t: Tys,
}

/// The fixture's types and callee, copied out so a body is built while the
/// module is borrowed by its Builder.
#[derive(Clone, Copy)]
struct Tys {
    int: Ty,
    bool: Ty,
    string: Ty,
    /// `&String`.
    string_ref: Ty,
    /// A pure function of one `String` to an `i64`.
    takes_string: Ty,
    /// A pure function of two `String`s to an `i64`.
    takes_two_strings: Ty,
    /// A pure function of one `i64` to an `i64`.
    takes_int: Ty,
    callee: QualifiedRef,
}

impl Tys {
    fn call(self, callee_ty: Ty, args: Vec<ValueId>) -> Op {
        Op::FunctionCall { callee: Callee::Direct(self.callee), callee_ty, args }
    }
}

fn fixture() -> Fixture {
    let mut types = Types::<Concrete>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let bool = types.intern(TyKind::Bool);
    let string = types.intern(TyKind::String);
    let string_ref = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(string)));
    let mut function = |params: &[Ty]| {
        types.intern(TyKind::Fn {
            params: params.into(),
            ret: int,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        })
    };
    let takes_string = function(&[string]);
    let takes_two_strings = function(&[string, string]);
    let takes_int = function(&[int]);
    let interner = Interner::new();
    let callee = QualifiedRef::root(interner.intern("f"));
    let t = Tys { int, bool, string, string_ref, takes_string, takes_two_strings, takes_int, callee };
    Fixture { module: Module::new(types), interner, t }
}

impl Fixture {
    fn context(&self, name: &str) -> QualifiedRef {
        QualifiedRef::root(self.interner.intern(name))
    }

    fn ops(&self, body: BodyId, block: BlockId) -> Vec<Op> {
        self.module.body(body).block(block).insts().map(|inst| inst.op().clone()).collect()
    }

    fn all_ops(&self, body: BodyId) -> Vec<Op> {
        self.module.body(body).blocks().flat_map(|(_, block)| block.insts()).map(|inst| inst.op().clone()).collect()
    }
}

/// Three contexts: `page`, an `i64` held in a slot; `clock`, a volatile
/// `i64`; and `note`, a `String` held in a slot.
fn declarations(interner: &Interner) -> Declarations {
    let mut types = Types::<crate::ty::Poly>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let string = types.intern(TyKind::String);
    let context =
        |name: &str, ty, volatile: bool| (QualifiedRef::root(interner.intern(name)), ContextDecl { ty, volatile });
    Declarations::verify(RawDeclarations {
        types,
        functions: Vec::new(),
        user_types: Vec::new(),
        casts: Vec::new(),
        conversions: Vec::new(),
        lang_impls: Vec::new(),
        contexts: vec![context("page", int, false), context("clock", int, true), context("note", string, false)],
        inputs: Vec::new(),
        loop_markers: LoopMarkers::default(),
        core: Core::of(interner),
    })
    .expect("three contexts under three names verify")
}

fn at(n: u32) -> Span {
    Span::new(n as usize, n as usize + 1)
}

fn text(t: Tys, builder: &mut Builder<'_>, span: Span) -> ValueId {
    builder.push(Op::Const { value: Literal::String("a".to_string()) }, t.string, span)
}

fn constant(n: i128) -> Op {
    Op::Const { value: Literal::Int(n) }
}

fn ret(value: ValueId) -> Terminator {
    Terminator::Return { value, order: None, span: Span::ZERO }
}

fn assign(slot: ValueId, value: ValueId) -> Op {
    Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }
}

fn take(slot: ValueId) -> Op {
    Op::Take { target: RefTarget::Var(slot), path: Vec::new(), taken_out: false }
}

fn commit(context: QualifiedRef, value: ValueId) -> Op {
    Op::Commit { context, value, wrote: true }
}

fn diamond(cond: ValueId, then_to: BlockId, else_to: BlockId, join: BlockId) -> Terminator {
    Terminator::Diamond { cond, then_to, then_args: Vec::new(), else_to, else_args: Vec::new(), join }
}

fn jump(to: BlockId, args: Vec<ValueId>) -> Terminator {
    Terminator::Jump { to, args }
}

fn run(f: &mut Fixture, step: fn(&mut Module, &mut AnalysisManager<'_>)) {
    let decls = declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    step(&mut f.module, &mut am);
}

fn arguments(op: &Op) -> &[ValueId] {
    match op {
        Op::FunctionCall { args, .. } => args,
        other => panic!("{other:?} is not a call"),
    }
}

fn is_clone_of(op: &Op, of: ValueId) -> bool {
    matches!(op, Op::StringClone { src } if *src == of)
}

fn is_drop_of(op: &Op, of: ValueId) -> bool {
    matches!(op, Op::Drop { src } if *src == of)
}

fn is_take_of(op: &Op, of: ValueId) -> bool {
    matches!(op, Op::Take { target: RefTarget::Var(slot), path, .. } if *slot == of && path.is_empty())
}

fn drops(ops: &[Op]) -> usize {
    ops.iter().filter(|op| matches!(op, Op::Drop { .. })).count()
}

// -- clone_strings -----------------------------------------------------------

/// `s = "a"; f(s); f(s)`: the first use is read again after it, so it takes
/// a clone of `s`, with the use's span; the second is the last and moves.
#[test]
fn a_string_moved_twice_is_cloned_at_the_first_use_and_moved_by_the_last() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let s = text(t, &mut builder, at(1));
    builder.push(t.call(t.takes_string, vec![s]), t.int, at(2));
    let last = builder.push(t.call(t.takes_string, vec![s]), t.int, at(3));
    builder.terminate(ret(last), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);

    let ops = f.ops(id, entry);
    let [_, clone, first, second] = &ops[..] else {
        panic!("one clone is added before the first call: {ops:?}")
    };
    assert!(is_clone_of(clone, s));
    let insts: Vec<_> = f.module.body(id).block(entry).insts().collect();
    assert_eq!(insts[1].span(), at(2), "the clone takes the span of the use it serves");
    assert_eq!(arguments(first), [insts[1].result().expect("a clone yields a value")]);
    assert_eq!(arguments(second), [s], "the last use moves the value itself");
}

/// `f(s, s)`: of the two operands that move `s`, the first takes a clone
/// and the last moves `s`.
#[test]
fn a_string_moved_twice_by_one_call_is_cloned_for_the_first_operand() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let s = text(t, &mut builder, at(1));
    let x = builder.push(t.call(t.takes_two_strings, vec![s, s]), t.int, at(2));
    builder.terminate(ret(x), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);

    let ops = f.ops(id, entry);
    let [_, clone, call] = &ops[..] else {
        panic!("one clone is added before the call: {ops:?}")
    };
    assert!(is_clone_of(clone, s));
    let cloned = f.module.body(id).block(entry).insts().nth(1).expect("the block holds that instruction").result().expect("a clone yields a value");
    assert_eq!(arguments(call), [cloned, s]);
}

/// `if c { f(s) } ; f(s)`: the arm's use is read again after the join, so
/// it takes a clone; the use after the join is the last.
#[test]
fn a_string_read_in_one_arm_and_after_the_join_is_cloned_in_the_arm() {
    let mut f = fixture();
    let t = f.t;
    let nc = f.interner.intern("c");
    let mut builder = f.module.build_body(false, t.int);
    let c = builder.param(nc, t.bool, acvus_ast::Span::ZERO);
    let s = text(t, &mut builder, at(1));
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join), at(7));
    builder.switch_to(then_block);
    builder.push(t.call(t.takes_string, vec![s]), t.int, at(2));
    builder.terminate(jump(join, Vec::new()), at(7));
    builder.switch_to(else_block);
    builder.terminate(jump(join, Vec::new()), at(7));
    builder.switch_to(join);
    let last = builder.push(t.call(t.takes_string, vec![s]), t.int, at(3));
    builder.terminate(ret(last), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);

    let arm = f.ops(id, then_block);
    let [clone, call] = &arm[..] else {
        panic!("the arm's use takes a clone: {arm:?}")
    };
    assert!(is_clone_of(clone, s));
    assert_ne!(arguments(call), [s]);
    let after = f.ops(id, join);
    let [call] = &after[..] else {
        panic!("the use after the join is not cloned: {after:?}")
    };
    assert_eq!(arguments(call), [s]);
    assert!(f.ops(id, else_block).is_empty());
}

/// An `i64` read twice and a `String` used once are never cloned.
#[test]
fn a_copying_value_and_a_last_use_are_never_cloned() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let n = builder.push(constant(1), t.int, at(1));
    builder.push(t.call(t.takes_int, vec![n]), t.int, at(2));
    builder.push(t.call(t.takes_int, vec![n]), t.int, at(3));
    let s = text(t, &mut builder, at(4));
    let last = builder.push(t.call(t.takes_string, vec![s]), t.int, at(5));
    builder.terminate(ret(last), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let before = f.all_ops(id).len();

    run(&mut f, clone_strings);

    assert_eq!(f.all_ops(id).len(), before);
    assert!(!f.all_ops(id).iter().any(|op| matches!(op, Op::StringClone { .. })));
}

/// `s` a `String` slot: `f(s); f(s)`. The first read is read again, so it
/// becomes a clone of `&s`; the second is the last and moves the content.
#[test]
fn a_string_slot_read_twice_is_cloned_at_the_first_read_and_moved_by_the_last() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let slot = builder.slot(t.string, at(1));
    let s = text(t, &mut builder, at(2));
    builder.push_unit(assign(slot, s), at(2));
    let first = builder.push(take(slot), t.string, at(3));
    builder.push(t.call(t.takes_string, vec![first]), t.int, at(3));
    let second = builder.push(take(slot), t.string, at(4));
    let last = builder.push(t.call(t.takes_string, vec![second]), t.int, at(4));
    builder.terminate(ret(last), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);

    let ops = f.ops(id, entry);
    let [_, _, reference, clone, first_call, moved, last_call] = &ops[..] else {
        panic!("the first read becomes a Ref and a clone: {ops:?}")
    };
    let insts: Vec<_> = f.module.body(id).block(entry).insts().collect();
    let (lent, cloned) = (insts[2].result(), insts[3].result());
    assert!(matches!(reference, Op::Ref { target: RefTarget::Var(named), mutability: Mutability::Shared, .. } if *named == slot));
    assert_eq!(f.module.body(id).ty_of(lent.expect("a Ref yields a value")), t.string_ref);
    assert!(is_clone_of(clone, lent.expect("a Ref yields a value")));
    assert_eq!((insts[2].span(), insts[3].span()), (at(3), at(3)), "the clone takes the Take's span");
    assert_eq!(arguments(first_call), [cloned.expect("a clone yields a value")]);
    assert!(is_take_of(moved, slot), "the last read moves the content");
    assert_eq!(arguments(last_call), [second]);
}

// -- remove_dead_context_ops -------------------------------------------------

/// `page` is loaded at entry and its slot overwritten before any read: the
/// load goes, its store with it. The slot is then written, so the commit
/// stays.
#[test]
fn an_entry_load_whose_slot_is_overwritten_before_any_read_is_removed() {
    let mut f = fixture();
    let t = f.t;
    let page = f.context("page");
    let mut builder = f.module.build_body(true, t.int);
    let entry = builder.current_block();
    let c = builder.context_slot(page, t.int, at(1));
    let loaded = builder.push_effect(Op::Fetch { context: page }, t.int, at(1));
    builder.push_unit(assign(c, loaded), at(1));
    let one = builder.push(constant(1), t.int, at(2));
    builder.push_unit(assign(c, one), at(2));
    let taken = builder.push(take(c), t.int, at(3));
    builder.push_effect_unit(commit(page, taken), at(3));
    builder.terminate(ret(one), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, remove_dead_context_ops);

    let ops = f.ops(id, entry);
    assert!(!ops.iter().any(|op| matches!(op, Op::Fetch { .. })), "the load goes: {ops:?}");
    let stored: Vec<ValueId> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Assign { value, .. } => Some(*value),
            _ => None,
        })
        .collect();
    assert_eq!(stored, vec![one], "the load's store goes, the program's stays");
    assert!(ops.iter().any(|op| matches!(op, Op::Commit { .. })), "the written context is committed");
}

/// `page` is loaded and read, and never written: its commit goes with the
/// `Take` that fed it, the content staying the page's, and the load, whose
/// value is read, stays.
#[test]
fn a_commit_of_a_context_never_written_is_removed_with_its_take() {
    let mut f = fixture();
    let t = f.t;
    let page = f.context("page");
    let mut builder = f.module.build_body(true, t.int);
    let entry = builder.current_block();
    let c = builder.context_slot(page, t.int, at(1));
    let loaded = builder.push_effect(Op::Fetch { context: page }, t.int, at(1));
    builder.push_unit(assign(c, loaded), at(1));
    let read = builder.push(take(c), t.int, at(2));
    let taken = builder.push(take(c), t.int, at(3));
    builder.push_effect_unit(commit(page, taken), at(3));
    builder.terminate(ret(read), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, remove_dead_context_ops);

    let ops = f.ops(id, entry);
    assert!(!ops.iter().any(|op| matches!(op, Op::Commit { .. })), "the commit goes: {ops:?}");
    assert_eq!(ops.iter().filter(|op| is_take_of(op, c)).count(), 1, "the commit's Take goes with it");
    assert!(ops.iter().any(|op| matches!(op, Op::Fetch { .. })), "the load is read and stays");
}

/// `clock` is volatile: a load nothing reads and a commit stay.
#[test]
fn a_volatile_load_and_commit_stay() {
    let mut f = fixture();
    let t = f.t;
    let clock = f.context("clock");
    let mut builder = f.module.build_body(true, t.int);
    let entry = builder.current_block();
    builder.push_effect(Op::Fetch { context: clock }, t.int, at(1));
    let one = builder.push(constant(1), t.int, at(2));
    builder.push_effect_unit(commit(clock, one), at(3));
    builder.terminate(ret(one), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let before = f.ops(id, entry).len();

    run(&mut f, remove_dead_context_ops);

    let ops = f.ops(id, entry);
    assert_eq!(ops.len(), before);
    assert!(ops.iter().any(|op| matches!(op, Op::Fetch { .. })));
    assert!(ops.iter().any(|op| matches!(op, Op::Commit { .. })));
}

// -- insert_drops ------------------------------------------------------------

/// `s` is read by a test and never again: it is dropped just after, with
/// that test's span.
#[test]
fn a_value_that_does_not_copy_is_dropped_after_its_last_use() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let s = text(t, &mut builder, at(1));
    builder.push(Op::TestLiteral { src: s, value: Literal::String("a".to_string()) }, t.bool, at(2));
    let n = builder.push(constant(1), t.int, at(3));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let [_, test, dropped, _] = &ops[..] else {
        panic!("one drop is added: {ops:?}")
    };
    assert!(matches!(test, Op::TestLiteral { .. }));
    assert!(is_drop_of(dropped, s));
    assert_eq!(f.module.body(id).block(entry).insts().nth(2).expect("the block holds that instruction").span(), at(2));
}

/// `f(s)` moves `s` to the callee, which drops it; the program does not.
#[test]
fn a_value_moved_into_a_call_is_not_dropped() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let s = text(t, &mut builder, at(1));
    let x = builder.push(t.call(t.takes_string, vec![s]), t.int, at(2));
    builder.terminate(ret(x), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    assert_eq!(drops(&f.all_ops(id)), 0);
}

/// `if c { f(s) } else { 0 }`: `s` is live into the then arm only. The then
/// arm moves it; on the edge into the else arm it ends, so that edge is
/// split and `s` is dropped at the new block's start, which the `Diamond`
/// now names as its else arm, with the `Diamond`'s span, where control
/// leaves.
#[test]
fn a_value_live_into_one_arm_only_is_dropped_at_the_other_arm_through_a_split_edge() {
    let mut f = fixture();
    let t = f.t;
    let nc = f.interner.intern("c");
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let c = builder.param(nc, t.bool, acvus_ast::Span::ZERO);
    let s = text(t, &mut builder, at(1));
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[t.int]);
    let p = builder.block_params(join)[0];
    builder.terminate(diamond(c, then_block, else_block, join), at(7));
    builder.switch_to(then_block);
    let x = builder.push(t.call(t.takes_string, vec![s]), t.int, at(2));
    builder.terminate(jump(join, vec![x]), at(8));
    builder.switch_to(else_block);
    let zero = builder.push(constant(0), t.int, at(3));
    builder.terminate(jump(join, vec![zero]), at(8));
    builder.switch_to(join);
    builder.terminate(ret(p), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let body = f.module.body(id);
    let Terminator::Diamond { then_to, else_to, .. } = body.block(entry).terminator() else {
        panic!("the entry still ends in its Diamond")
    };
    assert_eq!(*then_to, then_block, "the then edge ends nothing and is not split");
    assert_ne!(*else_to, else_block, "the else edge is split");
    let split = f.ops(id, *else_to);
    let [dropped] = &split[..] else {
        panic!("the split block holds the drop: {split:?}")
    };
    assert!(is_drop_of(dropped, s));
    assert_eq!(body.block(*else_to).insts().nth(0).expect("the block holds that instruction").span(), at(7), "an edge drop takes its source terminator's span");
    assert!(matches!(body.block(*else_to).terminator(), Terminator::Jump { to, .. } if *to == else_block));
    assert_eq!(drops(&f.all_ops(id)), 1, "s is dropped once, on the one path that does not move it");
}

/// `slot = "a"; slot = "b"; f(slot)`: the first content is stored and
/// never read, so it ends at its store, just before the overwrite, taken
/// out of the slot and dropped; the last read moves the second content.
#[test]
fn a_slot_content_overwritten_before_being_read_is_dropped_at_the_overwrite() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let slot = builder.slot(t.string, at(1));
    let first = text(t, &mut builder, at(1));
    let second = text(t, &mut builder, at(2));
    builder.push_unit(assign(slot, first), at(3));
    builder.push_unit(assign(slot, second), at(4));
    let read = builder.push(take(slot), t.string, at(5));
    let x = builder.push(t.call(t.takes_string, vec![read]), t.int, at(5));
    builder.terminate(ret(x), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let [_, _, _, taken, dropped, overwrite, _, _] = &ops[..] else {
        panic!("the first content is taken out and dropped before the overwrite: {ops:?}")
    };
    assert!(is_take_of(taken, slot));
    let content = f.module.body(id).block(entry).insts().nth(3).expect("the block holds that instruction").result().expect("a Take yields a value");
    assert!(is_drop_of(dropped, content));
    assert!(matches!(overwrite, Op::Assign { value, .. } if *value == second));
    assert_eq!(f.module.body(id).block(entry).insts().nth(4).expect("the block holds that instruction").span(), at(3), "the drop takes the ending store's span");
    assert_eq!(drops(&ops), 1, "the second content moves into the call");
}

/// `note` is loaded, then overwritten: the loaded content becomes the
/// body's at the overwrite and is dropped just before it. The final content
/// is committed, and nothing is dropped at the end.
#[test]
fn a_context_written_in_the_body_has_its_old_content_dropped_at_the_overwrite() {
    let mut f = fixture();
    let t = f.t;
    let note = f.context("note");
    let mut builder = f.module.build_body(true, t.int);
    let entry = builder.current_block();
    let c = builder.context_slot(note, t.string, at(1));
    let loaded = builder.push_effect(Op::Fetch { context: note }, t.string, at(1));
    builder.push_unit(assign(c, loaded), at(1));
    let written = text(t, &mut builder, at(2));
    builder.push_unit(assign(c, written), at(3));
    let taken = builder.push(take(c), t.string, at(4));
    builder.push_effect_unit(commit(note, taken), at(4));
    let n = builder.push(constant(0), t.int, at(5));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let [_, _, _, old, dropped, overwrite, _, _, _] = &ops[..] else {
        panic!("the old content is taken out and dropped before the overwrite: {ops:?}")
    };
    assert!(is_take_of(old, c));
    let content = f.module.body(id).block(entry).insts().nth(3).expect("the block holds that instruction").result().expect("a Take yields a value");
    assert!(is_drop_of(dropped, content));
    assert!(matches!(overwrite, Op::Assign { value, .. } if *value == written));
    assert_eq!(f.module.body(id).block(entry).insts().nth(4).expect("the block holds that instruction").span(), at(3), "the drop takes the overwrite's span");
    assert_eq!(drops(&ops), 1);
}

/// `note` is loaded and read through a reference, never written: its
/// content stays the page's, and the body drops nothing.
#[test]
fn a_context_never_written_gets_no_drop() {
    let mut f = fixture();
    let t = f.t;
    let note = f.context("note");
    let mut builder = f.module.build_body(true, t.bool);
    let entry = builder.current_block();
    let c = builder.context_slot(note, t.string, at(1));
    let loaded = builder.push_effect(Op::Fetch { context: note }, t.string, at(1));
    builder.push_unit(assign(c, loaded), at(1));
    let lent = builder.push(Op::Ref { target: RefTarget::Var(c), path: Vec::new(), mutability: Mutability::Shared }, t.string_ref, at(2));
    let same = builder.push(Op::StringEq { a: lent, b: lent }, t.bool, at(2));
    builder.terminate(ret(same), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let before = f.ops(id, entry).len();

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    assert_eq!(ops.len(), before, "nothing is added: {ops:?}");
    assert_eq!(drops(&ops), 0);
}

/// Integers and booleans, read and left unused, get no drop.
#[test]
fn copying_values_get_no_drop() {
    let mut f = fixture();
    let t = f.t;
    let nc = f.interner.intern("c");
    let mut builder = f.module.build_body(false, t.int);
    let c = builder.param(nc, t.bool, acvus_ast::Span::ZERO);
    let n = builder.push(constant(1), t.int, at(1));
    builder.push(constant(2), t.int, at(2));
    builder.push(t.call(t.takes_int, vec![n]), t.int, at(3));
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[]);
    builder.terminate(diamond(c, then_block, else_block, join), at(7));
    builder.switch_to(then_block);
    builder.terminate(jump(join, Vec::new()), at(7));
    builder.switch_to(else_block);
    builder.terminate(jump(join, Vec::new()), at(7));
    builder.switch_to(join);
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let blocks = f.module.body(id).blocks().count();

    run(&mut f, insert_drops);

    assert_eq!(drops(&f.all_ops(id)), 0);
    assert_eq!(f.module.body(id).blocks().count(), blocks, "no edge is split");
}

/// `if c { f(s); jump join(u) } else { jump join(s) }` where `join`'s
/// parameter is never read: the parameter goes with its arguments, so `s`
/// ends on the edge into `join`, which is split for its drop with the span
/// of the entry's terminator.
#[test]
fn a_dead_block_parameter_is_removed_and_its_argument_dropped_on_the_edge() {
    let mut f = fixture();
    let t = f.t;
    let nc = f.interner.intern("c");
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let c = builder.param(nc, t.bool, at(1));
    let s = text(t, &mut builder, at(2));
    let then_block = builder.new_block(&[]);
    let join = builder.new_block(&[t.string]);
    builder.terminate(
        Terminator::JumpIf { cond: c, then_to: then_block, then_args: Vec::new(), else_to: join, else_args: vec![s] },
        at(7),
    );
    builder.switch_to(then_block);
    builder.push(t.call(t.takes_string, vec![s]), t.int, at(3));
    let u = text(t, &mut builder, at(4));
    builder.terminate(jump(join, vec![u]), at(8));
    builder.switch_to(join);
    let n = builder.push(constant(0), t.int, at(5));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let body = f.module.body(id);
    assert!(body.block(join).params().is_empty(), "the unread parameter is removed");
    let Terminator::JumpIf { then_to, else_to, else_args, .. } = body.block(entry).terminator() else {
        panic!("the entry still ends in its branch")
    };
    assert_eq!(*then_to, then_block);
    assert!(else_args.is_empty(), "the argument goes with the parameter");
    let split = f.ops(id, *else_to);
    let [dropped] = &split[..] else {
        panic!("the split edge holds the drop of s: {split:?}")
    };
    assert!(is_drop_of(dropped, s));
    assert_eq!(body.block(*else_to).insts().nth(0).expect("the block holds that instruction").span(), at(7), "the drop takes the source terminator's span");
    assert!(f.ops(id, then_block).iter().any(|op| is_drop_of(op, u)), "u, now unread, ends where it is made");
}

/// A `String` parameter nothing reads is dropped at the entry's start,
/// with its binder's span.
#[test]
fn a_dead_body_parameter_is_dropped_at_the_entry_with_its_binders_span() {
    let mut f = fixture();
    let t = f.t;
    let ns = f.interner.intern("s");
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let s = builder.param(ns, t.string, at(4));
    let n = builder.push(constant(0), t.int, at(5));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let [dropped, _] = &ops[..] else {
        panic!("the parameter is dropped first: {ops:?}")
    };
    assert!(is_drop_of(dropped, s));
    assert_eq!(f.module.body(id).block(entry).insts().nth(0).expect("the block holds that instruction").span(), at(4));
}

/// `note` is loaded; the then arm takes it out and puts a new content back,
/// the else arm overwrites it, and the join overwrites it again. The then
/// arm's store fills an empty slot and drops nothing; the else arm's and the
/// join's each drop the content they replace.
#[test]
fn a_context_moved_out_on_one_arm_and_reassigned_on_both_is_dropped_at_each_overwrite_of_a_held_content() {
    let mut f = fixture();
    let t = f.t;
    let (note, nc) = (f.context("note"), f.interner.intern("c"));
    let mut builder = f.module.build_body(true, t.int);
    let cond = builder.param(nc, t.bool, at(1));
    let c = builder.context_slot(note, t.string, at(1));
    let loaded = builder.push_effect(Op::Fetch { context: note }, t.string, at(1));
    builder.push_unit(assign(c, loaded), at(1));
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(cond, then_block, else_block, join), at(7));
    builder.switch_to(then_block);
    let out = builder.push(take(c), t.string, at(2));
    builder.push(t.call(t.takes_string, vec![out]), t.int, at(2));
    let back = text(t, &mut builder, at(3));
    builder.push_unit(assign(c, back), at(3));
    builder.terminate(jump(join, Vec::new()), at(8));
    builder.switch_to(else_block);
    let other = text(t, &mut builder, at(4));
    builder.push_unit(assign(c, other), at(4));
    builder.terminate(jump(join, Vec::new()), at(9));
    builder.switch_to(join);
    let last = text(t, &mut builder, at(5));
    builder.push_unit(assign(c, last), at(5));
    let taken = builder.push(take(c), t.string, at(6));
    builder.push_effect_unit(commit(note, taken), at(6));
    let n = builder.push(constant(0), t.int, at(6));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    assert_eq!(drops(&f.ops(id, then_block)), 0, "the then arm refills an empty slot");
    let dropped_before = |block: BlockId, stored: ValueId| {
        let ops = f.ops(id, block);
        let store = ops.iter().position(|op| matches!(op, Op::Assign { value, .. } if *value == stored));
        let store = store.expect("the overwrite stays");
        store >= 2 && is_take_of(&ops[store - 2], c) && matches!(ops[store - 1], Op::Drop { .. })
    };
    assert!(dropped_before(else_block, other), "the loaded content is dropped before the else arm's overwrite");
    assert!(dropped_before(join, last), "the arms' content is dropped before the join's overwrite");
    assert_eq!(drops(&f.all_ops(id)), 2);
}

/// `note` is loaded; the then arm takes it out, the else arm leaves it, and
/// the join overwrites it. At the overwrite the slot holds content on the
/// else path only: that content's liveness ended on the edge from the else
/// arm, so it is dropped there, with that arm's terminator's span.
#[test]
fn a_context_held_on_one_path_only_at_an_overwrite_is_dropped_on_the_edge_where_it_is_held() {
    let mut f = fixture();
    let t = f.t;
    let (note, nc) = (f.context("note"), f.interner.intern("c"));
    let mut builder = f.module.build_body(true, t.int);
    let cond = builder.param(nc, t.bool, at(1));
    let c = builder.context_slot(note, t.string, at(1));
    let loaded = builder.push_effect(Op::Fetch { context: note }, t.string, at(1));
    builder.push_unit(assign(c, loaded), at(1));
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(cond, then_block, else_block, join), at(7));
    builder.switch_to(then_block);
    let out = builder.push(take(c), t.string, at(2));
    builder.push(t.call(t.takes_string, vec![out]), t.int, at(2));
    builder.terminate(jump(join, Vec::new()), at(8));
    builder.switch_to(else_block);
    builder.terminate(jump(join, Vec::new()), at(9));
    builder.switch_to(join);
    let last = text(t, &mut builder, at(5));
    builder.push_unit(assign(c, last), at(5));
    let taken = builder.push(take(c), t.string, at(6));
    builder.push_effect_unit(commit(note, taken), at(6));
    let n = builder.push(constant(0), t.int, at(6));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let body = f.module.body(id);
    assert!(matches!(body.block(then_block).terminator(), Terminator::Jump { to, .. } if *to == join), "the then edge is not split");
    let Terminator::Jump { to: middle, .. } = body.block(else_block).terminator() else {
        panic!("the else arm still jumps")
    };
    assert_ne!(*middle, join, "the else edge is split");
    let split = f.ops(id, *middle);
    let [taken_out, dropped] = &split[..] else {
        panic!("the split edge takes the content out and drops it: {split:?}")
    };
    assert!(is_take_of(taken_out, c));
    assert!(matches!(dropped, Op::Drop { .. }));
    assert_eq!(body.block(*middle).insts().nth(1).expect("the block holds that instruction").span(), at(9), "the drop takes the else arm's terminator's span");
    assert_eq!(drops(&f.all_ops(id)), 1);
}

/// A parameter lent by a `Ref` and also passed as a value lives in two
/// forms, which is lower's defect.
#[test]
#[should_panic(expected = "is named both as a storage and as a value")]
fn a_parameter_named_both_as_a_storage_and_as_a_value_is_refused() {
    let mut f = fixture();
    let t = f.t;
    let ns = f.interner.intern("s");
    let mut builder = f.module.build_body(false, t.int);
    let s = builder.param(ns, t.string, at(1));
    let lent = builder.push(Op::Ref { target: RefTarget::Param(s), path: Vec::new(), mutability: Mutability::Shared }, t.string_ref, at(2));
    builder.push(Op::StringEq { a: lent, b: lent }, t.bool, at(2));
    let x = builder.push(t.call(t.takes_string, vec![s]), t.int, at(3));
    builder.terminate(ret(x), Span::ZERO);
    builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);
}

// -- insert_drops: partial moves (ownership §6) --------------------------------

/// A structural object of two `String` fields `a` and `b`, and a pure
/// function of one `&String` to an `i64`.
fn two_strings(interner: &Interner, t: Tys, builder: &mut Builder<'_>) -> (Ty, Ty, acvus_utils::Astr, acvus_utils::Astr) {
    let (a, b) = (interner.intern("a"), interner.intern("b"));
    let mut fields = vec![(a, t.string), (b, t.string)];
    fields.sort_by_key(|(name, _)| *name);
    let record = builder.intern(TyKind::Object(crate::ty::ObjectTy::Structural(fields.into_boxed_slice(), ())));
    let reads_string = builder.intern(TyKind::Fn {
        params: Box::new([t.string_ref]),
        ret: t.int,
        effect: EffectTerm::Known(Effect::pure()),
        flows: FlowTerm::Known(Flows::none()),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    (record, reads_string, a, b)
}

fn part_take(slot: ValueId, field: acvus_utils::Astr) -> Op {
    Op::Take { target: RefTarget::Var(slot), path: vec![PathSeg::Field(field)], taken_out: false }
}

/// `s = { a, b }; f(s.a); g(&s.b)`: `s.a` moves out, and its holder, the
/// callee, ends it. `s.b` is last read through the reference `g` takes, so
/// it is taken out of `s` and dropped just after that call, with its span;
/// `s` is never dropped whole, and the part moved out is not dropped again.
#[test]
fn a_record_with_one_string_field_moved_drops_the_other_at_its_last_read() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let (record, reads_string, a, b) = two_strings(&f.interner, t, &mut builder);
    let entry = builder.current_block();
    let slot = builder.slot(record, at(1));
    let first = text(t, &mut builder, at(1));
    let second = text(t, &mut builder, at(1));
    let object = builder.push(Op::MakeObject { fields: vec![(a, first), (b, second)] }, record, at(1));
    builder.push_unit(assign(slot, object), at(1));
    let moved = builder.push(part_take(slot, a), t.string, at(2));
    builder.push(t.call(t.takes_string, vec![moved]), t.int, at(2));
    let lent = builder.push(
        Op::Ref { target: RefTarget::Var(slot), path: vec![PathSeg::Field(b)], mutability: Mutability::Shared },
        t.string_ref,
        at(3),
    );
    let read = builder.push(t.call(reads_string, vec![lent]), t.int, at(4));
    builder.terminate(ret(read), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let [.., last_call, taken, dropped] = &ops[..] else {
        panic!("the remaining part is taken out and dropped after its last read: {ops:?}")
    };
    assert_eq!(arguments(last_call), [lent]);
    assert!(
        matches!(taken, Op::Take { target: RefTarget::Var(named), path, .. } if *named == slot && *path == [PathSeg::Field(b)]),
        "{taken:?} takes out `s.b`"
    );
    let insts: Vec<_> = f.module.body(id).block(entry).insts().collect();
    let part = insts[ops.len() - 2].result().expect("a Take yields a value");
    assert!(is_drop_of(dropped, part));
    assert_eq!(insts[ops.len() - 1].span(), at(4), "the drop takes its last read's span");
    assert_eq!(drops(&ops), 1, "`s.a` moved to the callee and `s` is never dropped whole");
}

/// `t = (x, y); f(t.0)`: `t.0` moves to the callee; `t.1` is never read,
/// so its liveness ends at `t`'s definition, where it is taken out of `t`
/// and dropped. `t` itself is never dropped.
#[test]
fn a_tuple_with_one_part_moved_drops_the_other_where_its_liveness_ends() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let pair = builder.intern(TyKind::Tuple(Box::new([t.string, t.string]), ()));
    let entry = builder.current_block();
    let first = text(t, &mut builder, at(1));
    let second = text(t, &mut builder, at(1));
    let tuple = builder.push(Op::MakeTuple { elements: vec![first, second] }, pair, at(2));
    let moved = builder.push(Op::TupleIndex { tuple, index: 0 }, t.string, at(3));
    let x = builder.push(t.call(t.takes_string, vec![moved]), t.int, at(3));
    builder.terminate(ret(x), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let [_, _, made, taken, dropped, first_part, call] = &ops[..] else {
        panic!("the unread part is taken out and dropped where the tuple is made: {ops:?}")
    };
    assert!(matches!(made, Op::MakeTuple { .. }));
    assert!(matches!(taken, Op::TupleIndex { tuple: whole, index: 1 } if *whole == tuple));
    let part = f.module.body(id).block(entry).insts().nth(3).expect("the block holds that instruction").result().expect("an extraction yields a value");
    assert!(is_drop_of(dropped, part));
    assert_eq!(f.module.body(id).block(entry).insts().nth(4).expect("the block holds that instruction").span(), at(2), "the drop takes the definition's span");
    assert!(matches!(first_part, Op::TupleIndex { index: 0, .. }));
    assert_eq!(arguments(call), [moved]);
    assert_eq!(drops(&ops), 1, "`t.0` moved to the callee and `t` is never dropped whole");
}

/// `s = { a, b }; s.a = z; f(s.a); g(&s.b)`: the store into `s.a` replaces
/// the first `a`, which nothing reads, so it ends at the store that made
/// it; the second `a` moves to `f`, and `b` ends after its last read.
#[test]
fn a_part_replaced_before_any_read_is_dropped_where_it_was_stored() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let (record, reads_string, a, b) = two_strings(&f.interner, t, &mut builder);
    let entry = builder.current_block();
    let slot = builder.slot(record, at(1));
    let first = text(t, &mut builder, at(1));
    let second = text(t, &mut builder, at(1));
    let object = builder.push(Op::MakeObject { fields: vec![(a, first), (b, second)] }, record, at(1));
    builder.push_unit(assign(slot, object), at(1));
    let replacing = text(t, &mut builder, at(2));
    let into_a = Op::Assign { target: RefTarget::Var(slot), path: vec![PathSeg::Field(a)], value: replacing, restores: false };
    builder.push_unit(into_a, at(2));
    let moved = builder.push(part_take(slot, a), t.string, at(3));
    builder.push(t.call(t.takes_string, vec![moved]), t.int, at(3));
    let lent = builder.push(
        Op::Ref { target: RefTarget::Var(slot), path: vec![PathSeg::Field(b)], mutability: Mutability::Shared },
        t.string_ref,
        at(4),
    );
    let read = builder.push(t.call(reads_string, vec![lent]), t.int, at(4));
    builder.terminate(ret(read), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let takes_part = |op: &Op, field| {
        matches!(op, Op::Take { target: RefTarget::Var(named), path, .. } if *named == slot && *path == [PathSeg::Field(field)])
    };
    assert!(matches!(ops[3], Op::Assign { .. }), "the whole store: {ops:?}");
    assert!(takes_part(&ops[4], a) && matches!(ops[5], Op::Drop { .. }), "the first `a` ends at its store: {ops:?}");
    assert!(matches!(&ops[7], Op::Assign { path, .. } if *path == [PathSeg::Field(a)]), "then `s.a` is replaced: {ops:?}");
    let n = ops.len();
    assert!(takes_part(&ops[n - 2], b) && matches!(ops[n - 1], Op::Drop { .. }), "`b` ends after its last read: {ops:?}");
    assert_eq!(drops(&ops), 2);
}

/// D89, ownership §3: `f(s.a); f(s.b)`: `s.a` is not read again, so its
/// `Take` moves it though `s` is read after, and `s.b`'s last read moves
/// it; no read is a clone. Before D89 the first was a clone, since the
/// place as a whole was read again.
#[test]
fn a_string_part_is_cloned_only_where_that_part_is_read_again() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let (record, _, a, b) = two_strings(&f.interner, t, &mut builder);
    let slot = builder.slot(record, at(1));
    let first = text(t, &mut builder, at(1));
    let second = text(t, &mut builder, at(1));
    let object = builder.push(Op::MakeObject { fields: vec![(a, first), (b, second)] }, record, at(1));
    builder.push_unit(assign(slot, object), at(1));
    let moved = builder.push(part_take(slot, a), t.string, at(2));
    builder.push(t.call(t.takes_string, vec![moved]), t.int, at(2));
    let again = builder.push(part_take(slot, b), t.string, at(3));
    let last = builder.push(t.call(t.takes_string, vec![again]), t.int, at(3));
    builder.terminate(ret(last), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);

    let ops = f.all_ops(id);
    let takes = |field| {
        ops.iter()
            .filter(|op| matches!(op, Op::Take { target: RefTarget::Var(named), path, .. } if *named == slot && *path == [PathSeg::Field(field)]))
            .count()
    };
    assert_eq!(takes(a), 1, "`s.a` is not read again, so it moves: {ops:?}");
    assert_eq!(takes(b), 1, "`s.b`'s last read moves it: {ops:?}");
    assert!(!ops.iter().any(|op| matches!(op, Op::StringClone { .. })), "no part is read again after its move: {ops:?}");
}
