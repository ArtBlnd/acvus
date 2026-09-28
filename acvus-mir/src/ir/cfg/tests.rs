//! The core's invariants (core.rs), each test citing the one it checks.
//! Bodies are built through the Builder and edited through the Rewriter.

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, QualifiedRef};

use super::*;
use crate::ir::{BinOp, Callee, OrderEdge, Overflow, UnaryOp, ValueId};
use crate::ty::{
    Calls, Concrete, Contexts, Cost, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Reissue,
    SplitCost, Ty, TyKind, TypeArg, Types,
};

struct Fixture {
    module: Module,
    interner: Interner,
    int: Ty,
    bool: Ty,
    order: Ty,
    /// An effectful callee type that states it may be split.
    splittable: Ty,
    /// An effectful callee type that does not.
    whole: Ty,
    /// A pure callee type.
    pure: Ty,
}

fn fixture() -> Fixture {
    let mut types = Types::<Concrete>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let bool = types.intern(TyKind::Bool);
    let order = types.intern(TyKind::Order);
    let mut call_ty = |effect: Effect| {
        types.intern(TyKind::Fn {
            params: Box::new([]),
            ret: int,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        })
    };
    let opaque = |split| Effect::new(Reissue::Opaque, false, Contexts::new(), Contexts::new(), split);
    let splittable = call_ty(opaque(SplitCost::WithCost(Cost::stated(1))));
    let whole = call_ty(opaque(SplitCost::Unavailable));
    let pure = call_ty(Effect::pure());
    Fixture {
        module: Module::new(types, int, Flows::none()),
        interner: Interner::new(),
        int,
        bool,
        order,
        splittable,
        whole,
        pure,
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
}

fn build(module: &mut Module, ordered: bool) -> Builder<'_> {
    let mut builder = module.build_body(ordered);
    builder.set_span(Span::ZERO);
    builder
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

fn result_of(inst: &Inst) -> ValueId {
    inst.result().expect("the instruction yields a value")
}

fn edge_of(inst: &Inst) -> OrderEdge {
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

/// One block: `x = a + a; y = -x; return y`.
fn straight_line(f: &mut Fixture) -> (BodyId, ValueId, ValueId, ValueId) {
    let (name, int) = (f.name("a"), f.int);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(name, int);
    let x = builder.push(add(a, a), int);
    let y = builder.push(neg(x), int);
    builder.terminate(ret(y));
    (builder.seal().expect("the body has no open state"), a, x, y)
}

// -- 1. A value is made with its type, in the module's one table ---------

#[test]
fn every_value_is_born_typed() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(name, int);
    let slot = builder.slot(bool);
    let join = builder.new_block(&[bool]);
    let p = builder.block_params(join)[0];
    let t = builder.push(Op::Const { value: Literal::Bool(true) }, bool);
    builder.terminate(Terminator::Jump { to: join, args: vec![t] });
    builder.switch_to(join);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let body = f.body(id);
    assert_eq!(body.ty_of(a), int);
    assert_eq!(body.ty_of(p), bool);
    assert_eq!(body.ty_of(t), bool);
    assert_eq!(body.ty_of(slot), bool);
    assert_eq!(body.slots(), &[slot]);
}

#[test]
#[should_panic(expected = "types §4")]
fn a_reference_to_a_reference_is_refused() {
    let mut f = fixture();
    let (name, int) = (f.name("r"), f.int);
    let inner = f.module.types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    let outer = f.module.types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(inner)));
    build(&mut f.module, false).param(name, outer);
}

#[test]
#[should_panic(expected = "read in table")]
fn a_type_of_another_table_is_refused() {
    let mut f = fixture();
    let name = f.name("a");
    let foreign = Types::<Concrete>::new().intern(TyKind::Bool);
    build(&mut f.module, false).param(name, foreign);
}

#[test]
#[should_panic(expected = "pushed by push_unit")]
fn an_operation_without_a_result_is_not_given_one() {
    let mut f = fixture();
    let (name, int) = (f.name("a"), f.int);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(name, int);
    builder.push(Op::Drop { src: a }, int);
}

// -- 2. Def-use; a used value is removed only with its replacement -------

#[test]
fn removing_a_used_value_is_refused() {
    let mut f = fixture();
    let (id, a, x, _) = straight_line(&mut f);
    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(x), Err(StillUsed { value: x }));
    assert_eq!(rewriter.body().use_count(x), 1);
    assert_eq!(rewriter.body().use_count(a), 2);
    assert!(!rewriter.finish().touched(Aspect::Instructions));
}

#[test]
fn a_replaced_value_is_removed_and_its_operands_lose_their_uses() {
    let mut f = fixture();
    let (id, a, x, y) = straight_line(&mut f);
    let mut rewriter = f.module.rewrite(id);
    rewriter.replace_all_uses(x, a);
    assert_eq!(rewriter.body().use_count(a), 3);
    assert_eq!(rewriter.remove(x), Ok(()));
    let body = rewriter.body();
    assert_eq!(body.use_count(a), 1);
    let insts = body.block(body.entry()).insts();
    assert_eq!(insts.len(), 1);
    assert_eq!(insts[0].result(), Some(y));
    assert!(matches!(insts[0].op(), Op::UnaryOp { operand, .. } if *operand == a));
    let changes = rewriter.finish();
    assert!(changes.touched(Aspect::Instructions));
    assert!(!changes.touched(Aspect::CfgShape));
}

#[test]
fn a_removal_moves_the_definitions_after_it() {
    let mut f = fixture();
    let (name, int) = (f.name("a"), f.int);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(name, int);
    let x = builder.push(add(a, a), int);
    let y = builder.push(neg(a), int);
    let z = builder.push(neg(y), int);
    builder.terminate(ret(z));
    let id = builder.seal().expect("the body has no open state");

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(x), Ok(()));
    // `y` now stands first: replacing it must find its use in `z`, which
    // moved up with it.
    rewriter.replace_all_uses(y, a);
    assert_eq!(rewriter.remove(y), Ok(()));
    let body = rewriter.body();
    let insts = body.block(body.entry()).insts();
    assert_eq!(insts[0].result(), Some(z));
    assert!(matches!(insts[0].op(), Op::UnaryOp { operand, .. } if *operand == a));
}

#[test]
#[should_panic(expected = "replaced only by a value of its type")]
fn a_use_is_replaced_only_by_a_value_of_its_type() {
    let mut f = fixture();
    let (na, nb, int, bool) = (f.name("a"), f.name("b"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(na, int);
    let b = builder.param(nb, bool);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    f.module.rewrite(id).replace_all_uses(a, b);
}

// -- 3. Block parameters agree with every edge's arguments ---------------

#[test]
#[should_panic(expected = "passes 0 arguments to 1 parameters")]
fn an_edge_short_of_arguments_is_refused() {
    let mut f = fixture();
    let int = f.int;
    let mut builder = build(&mut f.module, false);
    let join = builder.new_block(&[int]);
    builder.terminate(jump(join));
}

#[test]
#[should_panic(expected = "passes")]
fn an_edge_argument_of_another_type_is_refused() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(name, int);
    let join = builder.new_block(&[bool]);
    builder.terminate(Terminator::Jump { to: join, args: vec![a] });
}

#[test]
#[should_panic(expected = "passes")]
fn a_reserved_block_is_checked_against_its_edges_when_made() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(name, int);
    let join = builder.reserve_block();
    builder.terminate(Terminator::Jump { to: join, args: vec![a] });
    builder.make_block(join, &[bool]);
}

#[test]
#[should_panic(expected = "cannot go to")]
fn a_retarget_to_other_parameters_is_refused() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(name, int);
    let takes_int = builder.new_block(&[int]);
    let takes_bool = builder.new_block(&[bool]);
    builder.terminate(Terminator::Jump { to: takes_int, args: vec![a] });
    for block in [takes_int, takes_bool] {
        builder.switch_to(block);
        builder.terminate(ret(a));
    }
    let id = builder.seal().expect("the body has no open state");
    f.module.rewrite(id).retarget(takes_int, takes_bool);
}

#[test]
fn removing_a_block_parameter_removes_its_argument_on_every_edge() {
    let mut f = fixture();
    let (na, nb, nc, int, bool) = (f.name("a"), f.name("b"), f.name("c"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(na, int);
    let b = builder.param(nb, int);
    let c = builder.param(nc, bool);
    let then_block = builder.new_block(&[]);
    let else_block = builder.new_block(&[]);
    let join = builder.reserve_block();
    builder.terminate(diamond(c, then_block, else_block, join));
    for (block, arg) in [(then_block, a), (else_block, b)] {
        builder.switch_to(block);
        builder.terminate(Terminator::Jump { to: join, args: vec![arg] });
    }
    builder.make_block(join, &[int]);
    let p = builder.block_params(join)[0];
    builder.switch_to(join);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.body().use_count(a), 2);
    assert_eq!(rewriter.remove(p), Ok(()));
    let body = rewriter.body();
    assert!(body.block(join).params().is_empty());
    for block in [then_block, else_block] {
        assert!(matches!(body.block(block).terminator(), Terminator::Jump { args, .. } if args.is_empty()));
    }
    assert_eq!(body.use_count(a), 1);
    assert_eq!(body.use_count(b), 0);
}

// -- 4. The order token chain (effects §2-3) -----------------------------

#[test]
fn push_effect_threads_the_token_and_return_carries_the_current_one() {
    let mut f = fixture();
    let (int, whole, pure) = (f.int, f.whole, f.pure);
    let (first_call, pure_call, second_call) = (f.call(whole), f.call(pure), f.call(whole));
    let mut builder = build(&mut f.module, true);
    let first = builder.push_effect(first_call, int);
    builder.push(pure_call, int);
    let second = builder.push_effect(second_call, int);
    builder.terminate(ret(second));
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    assert_eq!(body.ty_of(o0), f.order);
    let insts = body.block(body.entry()).insts();
    assert_eq!(result_of(&insts[0]), first);
    let first_edge = edge_of(&insts[0]);
    assert_eq!(first_edge.before, o0);
    assert_eq!(insts[1].order(), None);
    let second_edge = edge_of(&insts[2]);
    assert_eq!(second_edge.before, first_edge.after);
    assert_eq!(returned_order(body, body.entry()), Some(second_edge.after));
}

#[test]
#[should_panic(expected = "issues no effect")]
fn a_pure_body_takes_no_effect() {
    let mut f = fixture();
    let (int, call) = (f.int, f.call(f.whole));
    build(&mut f.module, false).push_effect(call, int);
}

#[test]
#[should_panic(expected = "pushed by push_effect")]
fn an_effectful_call_is_not_pushed_as_pure() {
    let mut f = fixture();
    let (int, call) = (f.int, f.call(f.whole));
    build(&mut f.module, true).push(call, int);
}

#[test]
fn the_token_crosses_a_diamond_join() {
    let mut f = fixture();
    let (nc, int, bool, whole) = (f.name("c"), f.int, f.bool, f.whole);
    let (arm_call, join_call) = (f.call(whole), f.call(whole));
    let mut builder = build(&mut f.module, true);
    let c = builder.param(nc, bool);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join));
    builder.switch_to(then_block);
    builder.push_effect(arm_call, int);
    builder.terminate(jump(join));
    builder.switch_to(else_block);
    builder.terminate(jump(join));
    builder.switch_to(join);
    let joined = builder.push_effect(join_call, int);
    builder.terminate(ret(joined));
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    let then_after = edge_of(&body.block(then_block).insts()[0]).after;
    let [token] = body.block(join).params() else {
        panic!("the join takes the token its arms pass: {:?}", body.block(join).params())
    };
    let passed = |block: BlockId| match body.block(block).terminator() {
        Terminator::Jump { args, .. } => args.clone(),
        other => panic!("{other:?} is a jump"),
    };
    assert_eq!(passed(then_block), vec![then_after]);
    assert_eq!(passed(else_block), vec![o0]);
    let join_edge = edge_of(&body.block(join).insts()[0]);
    assert_eq!(join_edge.before, *token);
    assert_eq!(returned_order(body, join), Some(join_edge.after));
}

#[test]
fn a_join_no_arm_advances_takes_no_token_parameter() {
    let mut f = fixture();
    let (nc, int, bool, whole) = (f.name("c"), f.int, f.bool, f.whole);
    let join_call = f.call(whole);
    let mut builder = build(&mut f.module, true);
    let c = builder.param(nc, bool);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join));
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(jump(join));
    }
    builder.switch_to(join);
    let joined = builder.push_effect(join_call, int);
    builder.terminate(ret(joined));
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    assert!(body.block(join).params().is_empty());
    assert_eq!(edge_of(&body.block(join).insts()[0]).before, body.order_param().expect("ordered"));
}

#[test]
fn the_token_crosses_a_loop_back_edge() {
    let mut f = fixture();
    let (nc, int, bool, whole) = (f.name("c"), f.int, f.bool, f.whole);
    let body_call = f.call(whole);
    let mut builder = build(&mut f.module, true);
    let c = builder.param(nc, bool);
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header));
    builder.switch_to(header);
    builder.terminate(Terminator::While {
        cond: c,
        stages: Stages::lowered(loop_body),
        exit,
        exit_args: Vec::new(),
    });
    builder.switch_to(loop_body);
    builder.push_effect(body_call, int);
    builder.terminate(jump(header));
    builder.switch_to(exit);
    builder.terminate(ret(c));
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    let [token] = body.block(header).params() else {
        panic!("the header takes the token around the loop: {:?}", body.block(header).params())
    };
    let body_edge = edge_of(&body.block(loop_body).insts()[0]);
    assert_eq!(body_edge.before, *token);
    assert!(matches!(body.block(body.entry()).terminator(), Terminator::Jump { args, .. } if *args == [o0]));
    assert!(matches!(body.block(loop_body).terminator(), Terminator::Jump { args, .. } if *args == [body_edge.after]));
    assert!(body.block(loop_body).params().is_empty());
    assert!(body.block(exit).params().is_empty());
    assert_eq!(returned_order(body, exit), Some(*token));
}

#[test]
fn anyorder_gives_its_entry_token_inside_and_merges_on_close() {
    let mut f = fixture();
    let (int, whole) = (f.int, f.whole);
    let calls = [f.call(whole), f.call(whole), f.call(whole)];
    let [first, second, after] = calls;
    let mut builder = build(&mut f.module, true);
    builder.open_anyorder();
    builder.push_effect(first, int);
    builder.open_anyorder();
    builder.push_effect(second, int);
    builder.close_anyorder();
    builder.close_anyorder();
    let last = builder.push_effect(after, int);
    builder.terminate(ret(last));
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    let insts = body.block(body.entry()).insts();
    let (a1, a2) = (edge_of(&insts[0]), edge_of(&insts[2]));
    assert_eq!((a1.before, a2.before), (o0, o0));
    let merged = |inst: &Inst| match inst.op() {
        Op::Merge { orders } => (orders.clone(), result_of(inst)),
        other => panic!("{other:?} is a merge"),
    };
    let (orders, m1) = merged(&insts[1]);
    assert_eq!(orders, vec![o0, a1.after]);
    let (orders, m2) = merged(&insts[3]);
    assert_eq!(orders, vec![m1, a2.after]);
    assert_eq!(edge_of(&insts[4]).before, m2);
}

#[test]
fn anyorder_merges_on_an_edge_that_leaves_it() {
    let mut f = fixture();
    let (int, whole) = (f.int, f.whole);
    let (inside, outside) = (f.call(whole), f.call(whole));
    let mut builder = build(&mut f.module, true);
    let out = builder.reserve_block();
    builder.open_anyorder();
    builder.push_effect(inside, int);
    builder.terminate(jump(out));
    let rest = builder.new_block(&[]);
    builder.switch_to(rest);
    builder.close_anyorder();
    builder.terminate(Terminator::Diverge);
    builder.make_block(out, &[]);
    builder.switch_to(out);
    let last = builder.push_effect(outside, int);
    builder.terminate(ret(last));
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let insts = body.block(body.entry()).insts();
    let merge = result_of(&insts[1]);
    assert!(matches!(insts[1].op(), Op::Merge { .. }));
    assert_eq!(edge_of(&body.block(out).insts()[0]).before, merge);
}

#[test]
fn a_return_inside_anyorder_carries_the_merge() {
    let mut f = fixture();
    let (int, whole) = (f.int, f.whole);
    let inside = f.call(whole);
    let mut builder = build(&mut f.module, true);
    builder.open_anyorder();
    let value = builder.push_effect(inside, int);
    builder.terminate(ret(value));
    let rest = builder.new_block(&[]);
    builder.switch_to(rest);
    builder.close_anyorder();
    builder.terminate(Terminator::Diverge);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let merge = result_of(&body.block(body.entry()).insts()[1]);
    assert_eq!(returned_order(body, body.entry()), Some(merge));
}

#[test]
fn removing_an_effectful_call_rethreads_the_token_around_it() {
    let mut f = fixture();
    let (int, whole) = (f.int, f.whole);
    let (first_call, second_call) = (f.call(whole), f.call(whole));
    let mut builder = build(&mut f.module, true);
    let first = builder.push_effect(first_call, int);
    let second = builder.push_effect(second_call, int);
    builder.terminate(ret(second));
    let id = builder.seal().expect("the body has no open state");
    let o0 = f.body(id).order_param().expect("ordered");

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(first), Ok(()));
    let body = rewriter.body();
    let insts = body.block(body.entry()).insts();
    assert_eq!(insts.len(), 1);
    assert_eq!(result_of(&insts[0]), second);
    assert_eq!(edge_of(&insts[0]).before, o0);
    assert_eq!(body.use_count(o0), 1);
    assert!(rewriter.finish().touched(Aspect::OrderTokens));
}

// -- 5. Spawn and TakeEffect are made only together (effects §6) ---------

#[test]
fn split_call_makes_a_start_and_its_taking() {
    let mut f = fixture();
    let (int, splittable) = (f.int, f.splittable);
    let call_op = f.call(splittable);
    let mut builder = build(&mut f.module, true);
    let call = builder.push_effect(call_op, int);
    let user = builder.push(neg(call), int);
    builder.terminate(ret(user));
    let id = builder.seal().expect("the body has no open state");
    let entry = f.body(id).entry();
    let call_edge = edge_of(&f.body(id).block(entry).insts()[0]);

    let mut rewriter = f.module.rewrite(id);
    rewriter.split_call(call);
    let body = rewriter.body();
    let insts = body.block(entry).insts();
    let handle = result_of(&insts[0]);
    assert!(matches!(insts[0].op(), Op::Spawn { .. }));
    assert_eq!(insts[0].order(), Some(OrderUse::Start(call_edge.before)));
    assert!(matches!(insts[1].op(), Op::TakeEffect { src } if *src == handle));
    assert_eq!(insts[1].result(), Some(call));
    assert_eq!(insts[1].order(), Some(OrderUse::Take(call_edge.after)));
    assert!(matches!(insts[2].op(), Op::UnaryOp { operand, .. } if *operand == call));
    assert!(matches!(rewriter.types().kind(body.ty_of(handle)), TyKind::Handle(ret) if *ret == f.int));
    assert_eq!(body.use_count(handle), 1);
    let changes = rewriter.finish();
    assert!(changes.touched(Aspect::Instructions));
    assert!(changes.touched(Aspect::OrderTokens));
}

#[test]
fn removing_a_taking_removes_its_start_and_rethreads_the_token() {
    let mut f = fixture();
    let (int, splittable, whole) = (f.int, f.splittable, f.whole);
    let (call_op, later_op) = (f.call(splittable), f.call(whole));
    let mut builder = build(&mut f.module, true);
    let call = builder.push_effect(call_op, int);
    let user = builder.push(neg(call), int);
    let later = builder.push_effect(later_op, int);
    builder.terminate(ret(later));
    let id = builder.seal().expect("the body has no open state");
    let o0 = f.body(id).order_param().expect("ordered");

    let mut rewriter = f.module.rewrite(id);
    rewriter.split_call(call);
    assert_eq!(rewriter.remove(call), Err(StillUsed { value: call }));
    assert_eq!(rewriter.remove(user), Ok(()));
    assert_eq!(rewriter.remove(call), Ok(()));
    let body = rewriter.body();
    let insts = body.block(body.entry()).insts();
    assert_eq!(insts.len(), 1);
    assert_eq!(result_of(&insts[0]), later);
    assert_eq!(edge_of(&insts[0]).before, o0);
}

#[test]
#[should_panic(expected = "may be split")]
fn a_call_its_callee_does_not_let_split_is_refused() {
    let mut f = fixture();
    let (int, call_op) = (f.int, f.call(f.whole));
    let mut builder = build(&mut f.module, true);
    let call = builder.push_effect(call_op, int);
    builder.terminate(ret(call));
    let id = builder.seal().expect("the body has no open state");
    f.module.rewrite(id).split_call(call);
}

#[test]
#[should_panic(expected = "written only by the core")]
fn a_start_is_not_pushed_alone() {
    let mut f = fixture();
    let (int, splittable, name) = (f.int, f.splittable, f.name("f"));
    build(&mut f.module, false).push(
        Op::Spawn { callee: Callee::Direct(QualifiedRef::root(name)), callee_ty: splittable, args: Vec::new() },
        int,
    );
}

// -- 6. A Diamond's join and a loop's stages and exit follow every edit ---

#[test]
fn a_retarget_updates_a_diamonds_join() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let [then_block, else_block, join, new_join] = [(); 4].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join));
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(jump(join));
    }
    for block in [join, new_join] {
        builder.switch_to(block);
        builder.terminate(ret(a));
    }
    let id = builder.seal().expect("the body has no open state");

    let mut rewriter = f.module.rewrite(id);
    rewriter.retarget(join, new_join);
    let body = rewriter.body();
    let Terminator::Diamond { join: now, .. } = body.block(body.entry()).terminator() else {
        panic!("the entry still ends in its Diamond")
    };
    assert_eq!(*now, new_join);
    for block in [then_block, else_block] {
        assert!(matches!(body.block(block).terminator(), Terminator::Jump { to, .. } if *to == new_join));
    }
    assert!(rewriter.finish().touched(Aspect::CfgShape));
}

#[test]
fn a_retarget_updates_a_loops_stages_and_exit() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let header = builder.current_block();
    let [loop_body, exit, other_body, other_exit] = [(); 4].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::While { cond: c, stages: Stages::lowered(loop_body), exit, exit_args: Vec::new() });
    for block in [loop_body, other_body] {
        builder.switch_to(block);
        builder.terminate(jump(header));
    }
    for block in [exit, other_exit] {
        builder.switch_to(block);
        builder.terminate(ret(a));
    }
    let id = builder.seal().expect("the body has no open state");

    let mut rewriter = f.module.rewrite(id);
    rewriter.retarget(loop_body, other_body);
    rewriter.retarget(exit, other_exit);
    let Terminator::While { stages, exit: now, .. } = rewriter.body().block(header).terminator() else {
        panic!("the header still ends in its loop")
    };
    assert_eq!(stages.body(), other_body);
    assert_eq!(*now, other_exit);
}

#[test]
fn a_split_keeps_the_block_named_and_moves_the_diamond_with_the_tail() {
    let mut f = fixture();
    let (na, nc, nd, int, bool) = (f.name("a"), f.name("c"), f.name("d"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let d = builder.param(nd, bool);
    let entry = builder.current_block();
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    let x = builder.push(add(a, a), int);
    let y = builder.push(neg(x), int);
    builder.terminate(diamond(c, then_block, else_block, join));
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(jump(join));
    }
    builder.switch_to(join);
    builder.terminate(ret(y));
    let id = builder.seal().expect("the body has no open state");

    let mut rewriter = f.module.rewrite(id);
    let tail = rewriter.split_block(entry, 1);
    let body = rewriter.body();
    assert!(matches!(body.block(entry).terminator(), Terminator::Jump { to, args } if *to == tail && args.is_empty()));
    assert!(matches!(body.block(tail).terminator(), Terminator::Diamond { join: j, .. } if *j == join));
    assert_eq!(body.block(tail).insts().len(), 1);
    // The moved instruction's definition and uses moved with it.
    rewriter.replace_all_uses(x, a);
    assert_eq!(rewriter.remove(x), Ok(()));
    let moved = &rewriter.body().block(tail).insts()[0];
    assert_eq!(moved.result(), Some(y));
    assert!(matches!(moved.op(), Op::UnaryOp { operand, .. } if *operand == a));
    // The Diamond's use of `c` moved with it to the tail's terminator.
    rewriter.replace_all_uses(c, d);
    assert!(matches!(rewriter.body().block(tail).terminator(), Terminator::Diamond { cond, .. } if *cond == d));
    assert_eq!(rewriter.body().use_count(c), 0);
}

// -- Builder: seal reports each kind of open state -----------------------

#[test]
fn seal_reports_a_block_without_its_terminator() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, false);
    let open = builder.new_block(&[]);
    builder.terminate(Terminator::Diverge);
    assert_eq!(builder.seal().err(), Some(Unsealed::BlockWithoutTerminator(open)));
}

#[test]
fn seal_reports_an_edge_to_an_unmade_block() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, false);
    let unmade = builder.reserve_block();
    builder.terminate(jump(unmade));
    assert_eq!(builder.seal().err(), Some(Unsealed::EdgeToUnmadeBlock(unmade)));
}

#[test]
fn seal_reports_a_reserved_block_nothing_names() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, false);
    let unmade = builder.reserve_block();
    builder.terminate(Terminator::Diverge);
    assert_eq!(builder.seal().err(), Some(Unsealed::ReservedNeverMade(unmade)));
}

#[test]
fn seal_reports_an_open_anyorder_scope() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, true);
    builder.open_anyorder();
    builder.terminate(Terminator::Diverge);
    assert_eq!(builder.seal().err(), Some(Unsealed::OpenAnyorder));
}

#[test]
fn a_sealed_body_is_added_to_the_module() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, false);
    builder.open_anyorder();
    builder.close_anyorder();
    builder.terminate(Terminator::Diverge);
    let id = builder.seal().expect("the body has no open state");
    f.module.set_entry(id);
    assert_eq!(f.module.entry(), id);
    assert!(matches!(f.body(id).block(f.body(id).entry()).terminator(), Terminator::Diverge));
}

#[test]
#[should_panic(expected = "no caller names an order token")]
fn a_return_does_not_take_a_token_from_its_caller() {
    let mut f = fixture();
    let (na, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false);
    let a = builder.param(na, int);
    let flag = builder.push(Op::Const { value: Literal::Bool(false) }, bool);
    builder.terminate(Terminator::Return { value: a, order: Some(flag), span: Span::ZERO });
}
