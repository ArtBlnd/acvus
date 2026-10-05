//! The required steps on bodies built with the Builder, each run through
//! its public entry over the whole module.

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, QualifiedRef};

use super::*;
use crate::decl::{ContextDecl, Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::cfg::edit::{DefDominatesUses, Invariant, OrderChainWhole, TypesAgree};
use crate::ir::cfg::{Builder, OrderUse};
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
    /// `&String`.
    string_ref: Ty,
    /// A pure function of one `String` to an `i64`.
    takes_string: Ty,
    /// A pure function of two `String`s to an `i64`.
    takes_two_strings: Ty,
    /// A pure function of one `i64` to an `i64`.
    takes_int: Ty,
    /// `&mut i64`.
    mut_int: Ty,
    /// A pure function of two `&mut i64`s to an `i64`.
    takes_two_mut_ints: Ty,
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
    let string_ref = types.intern(TyKind::Ref(Mutability::Shared, string));
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
    let mut_int = types.intern(TyKind::Ref(Mutability::Mut, int));
    let takes_two_mut_ints = types.intern(TyKind::Fn {
        params: Box::new([mut_int, mut_int]),
        ret: int,
        effect: EffectTerm::Known(Effect::pure()),
        flows: FlowTerm::Known(Flows::none()),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    let interner = Interner::new();
    let callee = QualifiedRef::root(interner.intern("f"));
    let t = Tys { int, bool, string, string_ref, takes_string, takes_two_strings, takes_int, mut_int, takes_two_mut_ints, callee };
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

/// Four contexts: `page`, an `i64`; `clock`, a volatile `i64`; `note`, a
/// `String`; and `tape`, a volatile `String`.
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
        generic_functions: Vec::new(),
        contexts: vec![
            context("page", int, false),
            context("clock", int, true),
            context("note", string, false),
            context("tape", string, true),
        ],
        entries: Vec::new(),
        loop_markers: LoopMarkers::default(),
        core: Core::of(interner),
    })
    .expect("four contexts under four names verify")
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

fn assign_context(context: QualifiedRef, value: ValueId) -> Op {
    Op::Assign { target: RefTarget::Context(context), path: Vec::new(), value, restores: false }
}

fn take_context(context: QualifiedRef) -> Op {
    Op::Take { target: RefTarget::Context(context), path: Vec::new(), taken_out: false }
}

fn is_context_take_of(op: &Op, context: QualifiedRef) -> bool {
    matches!(op, Op::Take { target: RefTarget::Context(named), path, .. } if *named == context && path.is_empty())
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

// -- reborrow_mut_uses -------------------------------------------------------

fn is_reborrow_of(op: &Op, of: ValueId) -> bool {
    matches!(op, Op::Ref { target: RefTarget::Through(reference), path, mutability: Mutability::Mut } if *reference == of && path.is_empty())
}

/// `x = 1; r = &mut x; f(r, r)`: each argument is a use of `r`, so each
/// takes a reborrow of its own, made just before the call with its span,
/// and `r` itself is passed nowhere (ownership §5).
#[test]
fn each_use_of_a_mut_reference_by_one_call_takes_a_reborrow_of_its_own() {
    let mut f = fixture();
    let t = f.t;
    let mut builder = f.module.build_body(false, t.int);
    let entry = builder.current_block();
    let x = builder.slot(t.int, at(1));
    let one = builder.push(constant(1), t.int, at(1));
    builder.push_unit(assign(x, one), at(1));
    let r = builder.push(Op::Ref { target: RefTarget::Var(x), path: Vec::new(), mutability: Mutability::Mut }, t.mut_int, at(2));
    let called = builder.push(t.call(t.takes_two_mut_ints, vec![r, r]), t.int, at(3));
    builder.terminate(ret(called), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, reborrow_mut_uses);

    let ops = f.ops(id, entry);
    let [_, _, _, first, second, call] = &ops[..] else {
        panic!("two reborrows are added before the call: {ops:?}")
    };
    assert!(is_reborrow_of(first, r) && is_reborrow_of(second, r));
    let insts: Vec<_> = f.module.body(id).block(entry).insts().collect();
    assert_eq!((insts[3].span(), insts[4].span()), (at(3), at(3)), "each reborrow takes the span of the use it serves");
    let made = |at: usize| insts[at].result().expect("a reborrow yields a value");
    assert_eq!(arguments(call), [made(3), made(4)]);
}

/// `(m: &mut i64) jump b1(m); b1(p): return p`: an edge's argument and a
/// returned value are uses of a `&mut`, each reborrowed at the end of its
/// block; a `Jump` has one edge, so the reborrow is made on it alone.
#[test]
fn a_mut_reference_passed_along_a_jump_and_returned_takes_a_reborrow_before_each_terminator() {
    let mut f = fixture();
    let t = f.t;
    let nm = f.interner.intern("m");
    let mut builder = f.module.build_body(false, t.mut_int);
    let entry = builder.current_block();
    let m = builder.param(nm, t.mut_int, Span::ZERO);
    let next = builder.new_block(&[t.mut_int]);
    let p = builder.block_params(next)[0];
    builder.terminate(jump(next, vec![m]), at(1));
    builder.switch_to(next);
    builder.terminate(ret(p), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, reborrow_mut_uses);

    let body = f.module.body(id);
    for (block, of) in [(entry, m), (next, p)] {
        let ops = f.ops(id, block);
        let [reborrow] = &ops[..] else { panic!("one reborrow ends {block:?}: {ops:?}") };
        assert!(is_reborrow_of(reborrow, of));
        let made = body.block(block).insts().next().expect("the reborrow").result().expect("a reborrow yields a value");
        assert_eq!(body.block(block).terminator().operands()[0], made, "the terminator takes the reborrow");
    }
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

/// `note` is overwritten: its content, the page's until then, becomes the
/// body's at the overwrite and is taken out and dropped just before it,
/// with the overwrite's span; nothing is dropped at the end.
#[test]
fn a_context_written_in_the_body_has_its_old_content_dropped_at_the_overwrite() {
    let mut f = fixture();
    let t = f.t;
    let note = f.context("note");
    let mut builder = f.module.build_body(true, t.int);
    let entry = builder.current_block();
    builder.context(note, t.string, false);
    let written = text(t, &mut builder, at(2));
    builder.push_unit(assign_context(note, written), at(3));
    let n = builder.push(constant(0), t.int, at(5));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    let [_, old, dropped, overwrite, _] = &ops[..] else {
        panic!("the old content is taken out and dropped before the overwrite: {ops:?}")
    };
    assert!(is_context_take_of(old, note));
    let content = f.module.body(id).block(entry).insts().nth(1).expect("the block holds that instruction").result().expect("a Take yields a value");
    assert!(is_drop_of(dropped, content));
    assert!(matches!(overwrite, Op::Assign { value, .. } if *value == written));
    assert_eq!(f.module.body(id).block(entry).insts().nth(2).expect("the block holds that instruction").span(), at(3), "the drop takes the overwrite's span");
    assert_eq!(drops(&ops), 1);
}

/// `note` is read through a reference, never written: its content stays
/// the page's, and the body drops nothing.
#[test]
fn a_context_never_written_gets_no_drop() {
    let mut f = fixture();
    let t = f.t;
    let note = f.context("note");
    let mut builder = f.module.build_body(true, t.bool);
    let entry = builder.current_block();
    builder.context(note, t.string, false);
    let lent = builder.push(Op::Ref { target: RefTarget::Context(note), path: Vec::new(), mutability: Mutability::Shared }, t.string_ref, at(2));
    let same = builder.push(Op::StringEq { a: lent, b: lent }, t.bool, at(2));
    builder.terminate(ret(same), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let before = f.ops(id, entry).len();

    run(&mut f, insert_drops);

    let ops = f.ops(id, entry);
    assert_eq!(ops.len(), before, "nothing is added: {ops:?}");
    assert_eq!(drops(&ops), 0);
}

/// `tape` is volatile and overwritten: the take that takes its old content
/// out before the overwrite stands on the order chain, just before the
/// overwrite (context §3), and the drop after it.
#[test]
fn an_overwrite_of_a_volatile_context_takes_its_old_content_out_on_the_chain() {
    let mut f = fixture();
    let t = f.t;
    let tape = f.context("tape");
    let mut builder = f.module.build_body(true, t.int);
    let entry = builder.current_block();
    builder.context(tape, t.string, true);
    let written = text(t, &mut builder, at(2));
    builder.push_effect_unit(assign_context(tape, written), at(3));
    let n = builder.push(constant(0), t.int, at(5));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let body = f.module.body(id);
    let insts: Vec<_> = body.block(entry).insts().collect();
    let [_, old, dropped, overwrite, _] = &insts[..] else {
        panic!("the old content is taken out and dropped before the overwrite: {:?}", f.ops(id, entry))
    };
    assert!(is_context_take_of(old.op(), tape));
    assert!(is_drop_of(dropped.op(), old.result().expect("a Take yields a value")));
    let (Some(OrderUse::Edge(take)), Some(OrderUse::Edge(store))) = (old.order(), overwrite.order()) else {
        panic!("the take and the overwrite stand on the chain: {:?} {:?}", old.order(), overwrite.order())
    };
    assert_eq!(store.before, take.after, "the take stands just before the overwrite on the chain");
    assert_eq!(dropped.order(), None);
    assert_eq!(<OrderChainWhole as Invariant>::check(body, f.module.types()), Ok(()));
}

/// `tape` is volatile: the then arm takes it out, the else arm leaves it,
/// and the join overwrites it. The content held on the else path only is
/// dropped on the split edge from the else arm, its take on the order chain
/// there, threaded into the token that edge passes on (context §3); the
/// overwrite stays on the chain and every invariant holds.
#[test]
fn a_volatile_context_held_on_one_path_only_at_an_overwrite_is_taken_out_on_the_chain_on_the_edge() {
    let mut f = fixture();
    let t = f.t;
    let (tape, nc) = (f.context("tape"), f.interner.intern("c"));
    let mut builder = f.module.build_body(true, t.int);
    let cond = builder.param(nc, t.bool, at(1));
    builder.context(tape, t.string, true);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(cond, then_block, else_block, join), at(7));
    builder.switch_to(then_block);
    let out = builder.push_effect(take_context(tape), t.string, at(2));
    builder.push(t.call(t.takes_string, vec![out]), t.int, at(2));
    builder.terminate(jump(join, Vec::new()), at(8));
    builder.switch_to(else_block);
    builder.terminate(jump(join, Vec::new()), at(9));
    builder.switch_to(join);
    let last = text(t, &mut builder, at(5));
    builder.push_effect_unit(assign_context(tape, last), at(5));
    let n = builder.push(constant(0), t.int, at(6));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    let body = f.module.body(id);
    let Terminator::Jump { to: middle, .. } = body.block(else_block).terminator() else {
        panic!("the else arm still jumps")
    };
    assert_ne!(*middle, join, "the else edge is split");
    let split: Vec<_> = body.block(*middle).insts().collect();
    let [taken_out, dropped] = &split[..] else {
        panic!("the split edge takes the content out and drops it: {:?}", f.ops(id, *middle))
    };
    assert!(is_context_take_of(taken_out.op(), tape));
    assert!(matches!(dropped.op(), Op::Drop { .. }));
    let Some(OrderUse::Edge(take)) = taken_out.order() else {
        panic!("the take out of the volatile context stands on the chain: {:?}", taken_out.order())
    };
    assert!(
        body.block(*middle).terminator().operands().contains(&take.after),
        "the split edge passes on the token the take yields: {:?}",
        body.block(*middle).terminator()
    );
    let store = body.block(join).insts().find(|inst| matches!(inst.op(), Op::Assign { .. })).expect("the overwrite stays");
    assert!(store.order().is_some(), "the overwrite stays on the chain");
    assert_eq!(drops(&f.all_ops(id)), 1);
    for check in [<OrderChainWhole as Invariant>::check, <TypesAgree as Invariant>::check, <DefDominatesUses as Invariant>::check] {
        assert_eq!(check(body, f.module.types()), Ok(()));
    }
}

/// `tape` is volatile and read twice, each read's content read again by
/// the return (context §1): each take becomes a clone of a shared lend of
/// it, the lend on the order chain in the take's place (context §3).
#[test]
fn a_volatile_string_context_read_twice_is_cloned_by_lends_on_the_chain() {
    let mut f = fixture();
    let t = f.t;
    let tape = f.context("tape");
    let mut builder = f.module.build_body(true, t.int);
    builder.context(tape, t.string, true);
    let first = builder.push_effect(take_context(tape), t.string, at(1));
    let second = builder.push_effect(take_context(tape), t.string, at(2));
    let n = builder.push(t.call(t.takes_two_strings, vec![first, second]), t.int, at(3));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, clone_strings);

    let body = f.module.body(id);
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let lends: Vec<_> = insts.iter().filter(|inst| matches!(inst.op(), Op::Ref { target: RefTarget::Context(named), .. } if *named == tape)).collect();
    assert_eq!(lends.len(), 2, "{:?}", f.all_ops(id));
    assert!(lends.iter().all(|inst| inst.order().is_some()), "each lend stands on the chain");
    assert_eq!(insts.iter().filter(|inst| is_context_take_of(inst.op(), tape)).count(), 0, "no take is left");
    assert_eq!(insts.iter().filter(|inst| matches!(inst.op(), Op::StringClone { .. })).count(), 2);
    assert_eq!(<OrderChainWhole as Invariant>::check(body, f.module.types()), Ok(()));
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

/// The then arm takes `note` out and puts a new content back, the else arm
/// overwrites it, and the join overwrites it again. The then arm's store
/// fills an empty context and drops nothing; the else arm's and the join's
/// each drop the content they replace.
#[test]
fn a_context_moved_out_on_one_arm_and_reassigned_on_both_is_dropped_at_each_overwrite_of_a_held_content() {
    let mut f = fixture();
    let t = f.t;
    let (note, nc) = (f.context("note"), f.interner.intern("c"));
    let mut builder = f.module.build_body(true, t.int);
    let cond = builder.param(nc, t.bool, at(1));
    builder.context(note, t.string, false);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(cond, then_block, else_block, join), at(7));
    builder.switch_to(then_block);
    let out = builder.push(take_context(note), t.string, at(2));
    builder.push(t.call(t.takes_string, vec![out]), t.int, at(2));
    let back = text(t, &mut builder, at(3));
    builder.push_unit(assign_context(note, back), at(3));
    builder.terminate(jump(join, Vec::new()), at(8));
    builder.switch_to(else_block);
    let other = text(t, &mut builder, at(4));
    builder.push_unit(assign_context(note, other), at(4));
    builder.terminate(jump(join, Vec::new()), at(9));
    builder.switch_to(join);
    let last = text(t, &mut builder, at(5));
    builder.push_unit(assign_context(note, last), at(5));
    let n = builder.push(constant(0), t.int, at(6));
    builder.terminate(ret(n), Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    run(&mut f, insert_drops);

    assert_eq!(drops(&f.ops(id, then_block)), 0, "the then arm refills an empty context");
    let dropped_before = |block: BlockId, stored: ValueId| {
        let ops = f.ops(id, block);
        let store = ops.iter().position(|op| matches!(op, Op::Assign { value, .. } if *value == stored));
        let store = store.expect("the overwrite stays");
        store >= 2 && is_context_take_of(&ops[store - 2], note) && matches!(ops[store - 1], Op::Drop { .. })
    };
    assert!(dropped_before(else_block, other), "the page's content is dropped before the else arm's overwrite");
    assert!(dropped_before(join, last), "the arms' content is dropped before the join's overwrite");
    assert_eq!(drops(&f.all_ops(id)), 2);
}

/// The then arm takes `note` out, the else arm leaves it, and the join
/// overwrites it. At the overwrite the context holds content on the else
/// path only: that content's liveness ended on the edge from the else arm,
/// so it is dropped there, with that arm's terminator's span.
#[test]
fn a_context_held_on_one_path_only_at_an_overwrite_is_dropped_on_the_edge_where_it_is_held() {
    let mut f = fixture();
    let t = f.t;
    let (note, nc) = (f.context("note"), f.interner.intern("c"));
    let mut builder = f.module.build_body(true, t.int);
    let cond = builder.param(nc, t.bool, at(1));
    builder.context(note, t.string, false);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(cond, then_block, else_block, join), at(7));
    builder.switch_to(then_block);
    let out = builder.push(take_context(note), t.string, at(2));
    builder.push(t.call(t.takes_string, vec![out]), t.int, at(2));
    builder.terminate(jump(join, Vec::new()), at(8));
    builder.switch_to(else_block);
    builder.terminate(jump(join, Vec::new()), at(9));
    builder.switch_to(join);
    let last = text(t, &mut builder, at(5));
    builder.push_unit(assign_context(note, last), at(5));
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
    assert!(is_context_take_of(taken_out, note));
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
    let record = builder.intern(TyKind::Object { fields: fields.into_boxed_slice(), label: () });
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
