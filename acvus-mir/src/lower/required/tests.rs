//! The required steps on bodies built with the Builder, each run through
//! its public entry over the whole module.

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, QualifiedRef};

use super::*;
use crate::decl::{ContextDecl, Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::cfg::Builder;
use crate::ty::{Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy};

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
    /// A pure function of one `String` to an `i64`.
    takes_string: Ty,
    /// A pure function of one `i64` to an `i64`.
    takes_int: Ty,
    callee: QualifiedRef,
}

impl Tys {
    fn call(self, callee_ty: Ty, arg: ValueId) -> Op {
        Op::FunctionCall { callee: Callee::Direct(self.callee), callee_ty, args: vec![arg] }
    }
}

fn fixture() -> Fixture {
    let mut types = Types::<Concrete>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let bool = types.intern(TyKind::Bool);
    let string = types.intern(TyKind::String);
    let mut function = |param: Ty| {
        types.intern(TyKind::Fn {
            params: Box::new([param]),
            ret: int,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        })
    };
    let takes_string = function(string);
    let takes_int = function(int);
    let interner = Interner::new();
    let callee = QualifiedRef::root(interner.intern("f"));
    Fixture { module: Module::new(types), interner, t: Tys { int, bool, string, takes_string, takes_int, callee } }
}

impl Fixture {
    fn context(&self, name: &str) -> QualifiedRef {
        QualifiedRef::root(self.interner.intern(name))
    }

    fn ops(&self, body: BodyId, block: BlockId) -> Vec<Op> {
        self.module.body(body).block(block).insts().iter().map(|inst| inst.op().clone()).collect()
    }

    fn all_ops(&self, body: BodyId) -> Vec<Op> {
        self.module.body(body).blocks().flat_map(|(_, block)| block.insts()).map(|inst| inst.op().clone()).collect()
    }
}

/// Two contexts of `i64`: `page`, held in a slot, and `clock`, volatile.
fn declarations(interner: &Interner) -> Declarations {
    let mut types = Types::<crate::ty::Poly>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let context = |name: &str, volatile: bool| (QualifiedRef::root(interner.intern(name)), ContextDecl { ty: int, volatile });
    Declarations::verify(RawDeclarations {
        types,
        functions: Vec::new(),
        user_types: Vec::new(),
        casts: Vec::new(),
        conversions: Vec::new(),
        contexts: vec![context("page", false), context("clock", true)],
        inputs: Vec::new(),
        loop_markers: LoopMarkers::default(),
        core: Core::of(interner),
    })
    .expect("two contexts under two names verify")
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

fn run(f: &mut Fixture, step: fn(&mut Module, &mut AnalysisManager<'_>)) {
    let decls = declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    step(&mut f.module, &mut am);
}

fn argument(op: &Op) -> ValueId {
    match op {
        Op::FunctionCall { args, .. } => args[0],
        other => panic!("{other:?} is not a call"),
    }
}

fn is_clone_of(op: &Op, of: ValueId) -> bool {
    matches!(op, Op::StringClone { src } if *src == of)
}

fn is_drop_of(op: &Op, of: ValueId) -> bool {
    matches!(op, Op::Drop { src } if *src == of)
}

// -- clone_strings -----------------------------------------------------------

/// `s = "a"; f(s); f(s)`: the first use is read again after it, so it takes
/// a clone of `s`; the second is the last, and moves `s`.
#[test]
fn a_string_moved_twice_is_cloned_at_the_first_use_and_moved_by_the_last() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let s = text(t, &mut builder, at(1));
    builder.push(t.call(t.takes_string, s), t.int, at(2));
    let last = builder.push(t.call(t.takes_string, s), t.int, at(3));
    builder.terminate(ret(last));
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);

    let ops = f.ops(id, entry);
    let [_, clone, first, second] = &ops[..] else {
        panic!("one clone is added before the first call: {ops:?}")
    };
    assert!(is_clone_of(clone, s));
    let insts = f.module.body(id).block(entry).insts();
    assert_eq!(insts[1].span(), at(2), "the clone takes the span of the use it serves");
    assert_eq!(argument(first), insts[1].result().expect("a clone yields a value"));
    assert_eq!(argument(second), s, "the last use moves the value itself");
}

/// `if c { f(s) } ; f(s)`: the arm's use is read again after the join, so
/// it takes a clone; the use after the join is the last.
#[test]
fn a_string_read_in_one_arm_and_after_the_join_is_cloned_in_the_arm() {
    let mut f = fixture();
    let t = f.t;
    let nc = f.interner.intern("c");
    let mut builder = f.module.build_body(false, t.int);
    let c = builder.param(nc, t.bool);
    let s = text(t, &mut builder, at(1));
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::Diamond {
        cond: c,
        then_to: then_block,
        then_args: Vec::new(),
        else_to: else_block,
        else_args: Vec::new(),
        join,
    });
    builder.switch_to(then_block);
    builder.push(t.call(t.takes_string, s), t.int, at(2));
    builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
    builder.switch_to(else_block);
    builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
    builder.switch_to(join);
    let last = builder.push(t.call(t.takes_string, s), t.int, at(3));
    builder.terminate(ret(last));
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);

    let arm = f.ops(id, then_block);
    let [clone, call] = &arm[..] else {
        panic!("the arm's use takes a clone: {arm:?}")
    };
    assert!(is_clone_of(clone, s));
    assert_ne!(argument(call), s);
    let after = f.ops(id, join);
    let [call] = &after[..] else {
        panic!("the use after the join is not cloned: {after:?}")
    };
    assert_eq!(argument(call), s);
    assert!(f.ops(id, else_block).is_empty());
}

/// An `i64` read twice and a `String` used once are never cloned.
#[test]
fn a_copying_value_and_a_last_use_are_never_cloned() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let n = builder.push(constant(1), t.int, at(1));
    builder.push(t.call(t.takes_int, n), t.int, at(2));
    builder.push(t.call(t.takes_int, n), t.int, at(3));
    let s = text(t, &mut builder, at(4));
    let last = builder.push(t.call(t.takes_string, s), t.int, at(5));
    builder.terminate(ret(last));
    let id = builder.seal().expect("the body has no open state");
    let before = f.all_ops(id).len();

    run(&mut f, clone_strings);

    assert_eq!(f.all_ops(id).len(), before);
    assert!(!f.all_ops(id).iter().any(|op| matches!(op, Op::StringClone { .. })));
}

/// `s` a `String` slot: `f(s); f(s)`. The first read is a clone and the
/// second moves; the step stops, since no editor operation clones a slot's
/// content in place of its `Take`.
#[test]
#[should_panic(expected = "no editor operation clones a slot's content in place of its Take")]
fn a_string_slot_read_twice_stops_where_a_slot_read_cannot_be_cloned() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let slot = builder.slot(t.string, at(1));
    let s = text(t, &mut builder, at(2));
    builder.push_unit(assign(slot, s), at(2));
    let first = builder.push(take(slot), t.string, at(3));
    builder.push(t.call(t.takes_string, first), t.int, at(3));
    let second = builder.push(take(slot), t.string, at(4));
    let last = builder.push(t.call(t.takes_string, second), t.int, at(4));
    builder.terminate(ret(last));
    builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);
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
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let c = builder.context_slot(page, t.int, at(1));
    let loaded = builder.push(Op::Fetch { context: page }, t.int, at(1));
    builder.push_unit(assign(c, loaded), at(1));
    let one = builder.push(constant(1), t.int, at(2));
    builder.push_unit(assign(c, one), at(2));
    let t = builder.push(take(c), t.int, at(3));
    builder.push_unit(commit(page, t), at(3));
    builder.terminate(ret(one));
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

/// `page` is loaded and read, and never written: its commit goes, and the
/// load, whose value is read, stays.
#[test]
fn a_commit_of_a_context_never_written_is_removed() {
    let mut f = fixture();
    let t = f.t;
    let page = f.context("page");
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let c = builder.context_slot(page, t.int, at(1));
    let loaded = builder.push(Op::Fetch { context: page }, t.int, at(1));
    builder.push_unit(assign(c, loaded), at(1));
    let read = builder.push(take(c), t.int, at(2));
    let t = builder.push(take(c), t.int, at(3));
    builder.push_unit(commit(page, t), at(3));
    builder.terminate(ret(read));
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, remove_dead_context_ops);

    let ops = f.ops(id, entry);
    assert!(!ops.iter().any(|op| matches!(op, Op::Commit { .. })), "the commit goes: {ops:?}");
    assert!(ops.iter().any(|op| matches!(op, Op::Fetch { .. })), "the load is read and stays");
}

/// `clock` is volatile: a load nothing reads and a commit stay.
#[test]
fn a_volatile_load_and_commit_stay() {
    let mut f = fixture();
    let t = f.t;
    let clock = f.context("clock");
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    builder.push(Op::Fetch { context: clock }, t.int, at(1));
    let one = builder.push(constant(1), t.int, at(2));
    builder.push_unit(commit(clock, one), at(3));
    builder.terminate(ret(one));
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
    builder.terminate(ret(n));
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let [_, test, dropped, _] = &ops[..] else {
        panic!("one drop is added: {ops:?}")
    };
    assert!(matches!(test, Op::TestLiteral { .. }));
    assert!(is_drop_of(dropped, s));
    assert_eq!(f.module.body(id).block(entry).insts()[2].span(), at(2));
}

/// `f(s)` moves `s` to the callee, which drops it; the program does not.
#[test]
fn a_value_moved_into_a_call_is_not_dropped() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let s = text(t, &mut builder, at(1));
    let x = builder.push(t.call(t.takes_string, s), t.int, at(2));
    builder.terminate(ret(x));
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    assert!(!f.all_ops(id).iter().any(|op| matches!(op, Op::Drop { .. })));
}

/// `if c { f(s) } else { 0 }`: `s` is live into the then arm only. The then
/// arm moves it; on the edge into the else arm it ends, so that edge is
/// split and `s` is dropped at the new block's start, which the `Diamond`
/// now names as its else arm.
#[test]
fn a_value_live_into_one_arm_only_is_dropped_at_the_other_arm_through_a_split_edge() {
    let mut f = fixture();
    let t = f.t;
    let nc = f.interner.intern("c");
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let c = builder.param(nc, t.bool);
    let s = text(t, &mut builder, at(1));
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[t.int]);
    let p = builder.block_params(join)[0];
    builder.terminate(Terminator::Diamond {
        cond: c,
        then_to: then_block,
        then_args: Vec::new(),
        else_to: else_block,
        else_args: Vec::new(),
        join,
    });
    builder.switch_to(then_block);
    let x = builder.push(t.call(t.takes_string, s), t.int, at(2));
    builder.terminate(Terminator::Jump { to: join, args: vec![x] });
    builder.switch_to(else_block);
    let zero = builder.push(constant(0), t.int, at(3));
    builder.terminate(Terminator::Jump { to: join, args: vec![zero] });
    builder.switch_to(join);
    builder.terminate(ret(p));
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
    assert_eq!(body.block(*else_to).insts()[0].span(), at(1), "an edge drop takes its value's defining span");
    assert!(matches!(body.block(*else_to).terminator(), Terminator::Jump { to, .. } if *to == else_block));
    let drops = f.all_ops(id).into_iter().filter(|op| matches!(op, Op::Drop { .. })).count();
    assert_eq!(drops, 1, "s is dropped once, on the one path that does not move it");
}

/// `slot = "a"; slot = "b"; f(slot)`: the first content is overwritten
/// before any read and ends at the overwrite. The step stops, since no
/// editor operation drops a slot's content: a drop of the slot just before
/// the store is refused, since the store names the slot.
#[test]
#[should_panic(expected = "no editor operation drops a slot's content")]
fn a_slot_content_overwritten_before_being_read_stops_where_no_operation_drops_it() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let slot = builder.slot(t.string, at(1));
    let first = text(t, &mut builder, at(1));
    builder.push_unit(assign(slot, first), at(1));
    let second = text(t, &mut builder, at(2));
    builder.push_unit(assign(slot, second), at(2));
    let read = builder.push(take(slot), t.string, at(3));
    let x = builder.push(t.call(t.takes_string, read), t.int, at(3));
    builder.terminate(ret(x));
    builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);
}

/// Integers and booleans, read and left unused, get no drop.
#[test]
fn copying_values_get_no_drop() {
    let mut f = fixture();
    let t = f.t;
    let nc = f.interner.intern("c");
    let mut builder = f.module.build_body(false, t.int);
    let c = builder.param(nc, t.bool);
    let n = builder.push(constant(1), t.int, at(1));
    builder.push(constant(2), t.int, at(2));
    builder.push(t.call(t.takes_int, n), t.int, at(3));
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[]);
    builder.terminate(Terminator::Diamond {
        cond: c,
        then_to: then_block,
        then_args: Vec::new(),
        else_to: else_block,
        else_args: Vec::new(),
        join,
    });
    builder.switch_to(then_block);
    builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
    builder.switch_to(else_block);
    builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
    builder.switch_to(join);
    builder.terminate(ret(n));
    let id = builder.seal().expect("the body has no open state");
    let blocks = f.module.body(id).blocks().count();

    run(&mut f, insert_drops);

    assert!(!f.all_ops(id).iter().any(|op| matches!(op, Op::Drop { .. })));
    assert_eq!(f.module.body(id).blocks().count(), blocks, "no edge is split");
}
