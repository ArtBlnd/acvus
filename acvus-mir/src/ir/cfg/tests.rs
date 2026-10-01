//! The core's invariants (core.rs), each test citing the one it checks.
//! Bodies are built through the Builder and edited through the Rewriter.

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, QualifiedRef};

use super::*;
use crate::ir::{BinOp, Callee, OrderEdge, RefTarget, SwitchKey, UnaryOp, ValueId};
use crate::ty::{
    Calls, Concrete, Contexts, Cost, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Reissue,
    SplitCost, Ty, TyKind, Types,
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
            values: (),
        })
    };
    let opaque = |split| Effect::new(Reissue::Opaque, false, Contexts::new(), Contexts::new(), split, false);
    let splittable = call_ty(opaque(SplitCost::WithCost(Cost::stated(1))));
    let whole = call_ty(opaque(SplitCost::Unavailable));
    let pure = call_ty(Effect::pure());
    Fixture {
        module: Module::new(types),
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

fn build(module: &mut Module, ordered: bool, ret: Ty) -> Builder<'_> {
    module.build_body(ordered, ret)
}

/// The span of the `n`th instruction a test writes, distinct so each test
/// can see that a record keeps the span it was pushed with.
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
    Op::BinOp { op: BinOp::Add, left, right }
}

fn neg(operand: ValueId) -> Op {
    Op::UnaryOp { op: UnaryOp::Neg, operand }
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
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    let x = builder.push(add(a, a), int, at(1));
    let y = builder.push(neg(x), int, at(2));
    builder.terminate(ret(y), acvus_ast::Span::ZERO);
    (builder.seal().expect("the body has no open state"), a, x, y)
}

// -- 1. A value is made with its type, in the module's one table ---------

#[test]
fn every_value_is_born_typed() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    let slot = builder.slot(bool, at(0));
    let join = builder.new_block(&[bool]);
    let p = builder.block_params(join)[0];
    let t = builder.push(Op::Const { value: Literal::Bool(true) }, bool, at(3));
    builder.terminate(Terminator::Jump { to: join, args: vec![t] }, acvus_ast::Span::ZERO);
    builder.switch_to(join);
    builder.terminate(ret(a), acvus_ast::Span::ZERO);
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
    let inner = f.module.types.intern(TyKind::Ref(Mutability::Shared, int));
    let outer = f.module.types.intern(TyKind::Ref(Mutability::Shared, inner));
    build(&mut f.module, false, f.int).param(name, outer, acvus_ast::Span::ZERO);
}

#[test]
#[should_panic(expected = "read in table")]
fn a_type_of_another_table_is_refused() {
    let mut f = fixture();
    let name = f.name("a");
    let foreign = Types::<Concrete>::new().intern(TyKind::Bool);
    build(&mut f.module, false, f.int).param(name, foreign, acvus_ast::Span::ZERO);
}

#[test]
#[should_panic(expected = "pushed by push_unit")]
fn an_operation_without_a_result_is_not_given_one() {
    let mut f = fixture();
    let (name, int) = (f.name("a"), f.int);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    builder.push(Op::Drop { src: a }, int, at(4));
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
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
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
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    let x = builder.push(add(a, a), int, at(5));
    let y = builder.push(neg(a), int, at(6));
    let z = builder.push(neg(y), int, at(7));
    builder.terminate(ret(z), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(x), Ok(()));
    // `y` now stands first: replacing it must find its use in `z`, which
    // moved up with it.
    rewriter.replace_all_uses(y, a);
    assert_eq!(rewriter.remove(y), Ok(()));
    let body = rewriter.body();
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
    assert_eq!(insts[0].result(), Some(z));
    assert!(matches!(insts[0].op(), Op::UnaryOp { operand, .. } if *operand == a));
}

#[test]
#[should_panic(expected = "replaced only by a value of its type")]
fn a_use_is_replaced_only_by_a_value_of_its_type() {
    let mut f = fixture();
    let (na, nb, int, bool) = (f.name("a"), f.name("b"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let b = builder.param(nb, bool, acvus_ast::Span::ZERO);
    builder.terminate(ret(a), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    f.module.rewrite(id).replace_all_uses(a, b);
}

// -- 3. Block parameters agree with every edge's arguments ---------------

#[test]
#[should_panic(expected = "passes 0 arguments to 1 parameters")]
fn an_edge_short_of_arguments_is_refused() {
    let mut f = fixture();
    let int = f.int;
    let mut builder = build(&mut f.module, false, f.int);
    let join = builder.new_block(&[int]);
    builder.terminate(jump(join), acvus_ast::Span::ZERO);
}

#[test]
#[should_panic(expected = "passes")]
fn an_edge_argument_of_another_type_is_refused() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    let join = builder.new_block(&[bool]);
    builder.terminate(Terminator::Jump { to: join, args: vec![a] }, acvus_ast::Span::ZERO);
}

#[test]
#[should_panic(expected = "passes")]
fn a_reserved_block_is_checked_against_its_edges_when_made() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    let join = builder.reserve_block();
    builder.terminate(Terminator::Jump { to: join, args: vec![a] }, acvus_ast::Span::ZERO);
    builder.make_block(join, &[bool]);
}

#[test]
#[should_panic(expected = "cannot go to")]
fn a_retarget_to_other_parameters_is_refused() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    let takes_int = builder.new_block(&[int]);
    let takes_bool = builder.new_block(&[bool]);
    builder.terminate(Terminator::Jump { to: takes_int, args: vec![a] }, acvus_ast::Span::ZERO);
    for block in [takes_int, takes_bool] {
        builder.switch_to(block);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
    }
    let id = builder.seal().expect("the body has no open state");
    f.module.rewrite(id).retarget(takes_int, takes_bool);
}

#[test]
fn removing_a_block_parameter_removes_its_argument_on_every_edge() {
    let mut f = fixture();
    let (na, nb, nc, int, bool) = (f.name("a"), f.name("b"), f.name("c"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let b = builder.param(nb, int, acvus_ast::Span::ZERO);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let then_block = builder.new_block(&[]);
    let else_block = builder.new_block(&[]);
    let join = builder.reserve_block();
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    for (block, arg) in [(then_block, a), (else_block, b)] {
        builder.switch_to(block);
        builder.terminate(Terminator::Jump { to: join, args: vec![arg] }, acvus_ast::Span::ZERO);
    }
    builder.make_block(join, &[int]);
    let p = builder.block_params(join)[0];
    builder.switch_to(join);
    builder.terminate(ret(a), acvus_ast::Span::ZERO);
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
    let mut builder = build(&mut f.module, true, f.int);
    let first = builder.push_effect(first_call, int, at(8));
    builder.push(pure_call, int, at(9));
    let second = builder.push_effect(second_call, int, at(10));
    builder.terminate(ret(second), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    assert_eq!(body.ty_of(o0), f.order);
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
    assert_eq!(result_of(&insts[0]), first);
    let first_edge = edge_of(&insts[0]);
    assert_eq!(first_edge.before, o0);
    assert_eq!(insts[1].order(), None);
    let second_edge = edge_of(&insts[2]);
    assert_eq!(second_edge.before, first_edge.after);
    assert_eq!(returned_order(body, body.entry()), Some(second_edge.after));
    let spans: Vec<Span> = insts.iter().map(|inst| inst.span()).collect();
    assert_eq!(spans, vec![at(8), at(9), at(10)]);
}

#[test]
fn every_record_keeps_the_span_it_was_pushed_with() {
    let mut f = fixture();
    let (name, int) = (f.name("a"), f.int);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(name, int, acvus_ast::Span::ZERO);
    let x = builder.push(neg(a), int, at(100));
    builder.push_unit(Op::Drop { src: x }, at(101));
    builder.terminate(ret(a), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let body = f.body(id);
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
    assert_eq!((insts[0].span(), insts[1].span()), (at(100), at(101)));
    assert_eq!(insts[1].result(), None);
}

#[test]
#[should_panic(expected = "issues no effect")]
fn a_pure_body_takes_no_effect() {
    let mut f = fixture();
    let (int, call) = (f.int, f.call(f.whole));
    build(&mut f.module, false, f.int).push_effect(call, int, at(11));
}

#[test]
#[should_panic(expected = "pushed by push_effect")]
fn an_effectful_call_is_not_pushed_as_pure() {
    let mut f = fixture();
    let (int, call) = (f.int, f.call(f.whole));
    build(&mut f.module, true, f.int).push(call, int, at(12));
}

#[test]
fn the_token_crosses_a_diamond_join() {
    let mut f = fixture();
    let (nc, int, bool, whole) = (f.name("c"), f.int, f.bool, f.whole);
    let (arm_call, join_call) = (f.call(whole), f.call(whole));
    let mut builder = build(&mut f.module, true, f.int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    builder.push_effect(arm_call, int, at(13));
    builder.terminate(jump(join), acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    builder.terminate(jump(join), acvus_ast::Span::ZERO);
    builder.switch_to(join);
    let joined = builder.push_effect(join_call, int, at(14));
    builder.terminate(ret(joined), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    let then_after = edge_of(body.block(then_block).insts().nth(0).expect("the block holds that instruction")).after;
    let [token] = body.block(join).params() else {
        panic!("the join takes the token its arms pass: {:?}", body.block(join).params())
    };
    let passed = |block: BlockId| match body.block(block).terminator() {
        Terminator::Jump { args, .. } => args.clone(),
        other => panic!("{other:?} is a jump"),
    };
    assert_eq!(passed(then_block), vec![then_after]);
    assert_eq!(passed(else_block), vec![o0]);
    let join_edge = edge_of(body.block(join).insts().nth(0).expect("the block holds that instruction"));
    assert_eq!(join_edge.before, *token);
    assert_eq!(returned_order(body, join), Some(join_edge.after));
}

#[test]
fn a_join_no_arm_advances_takes_no_token_parameter() {
    let mut f = fixture();
    let (nc, int, bool, whole) = (f.name("c"), f.int, f.bool, f.whole);
    let join_call = f.call(whole);
    let mut builder = build(&mut f.module, true, f.int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(jump(join), acvus_ast::Span::ZERO);
    }
    builder.switch_to(join);
    let joined = builder.push_effect(join_call, int, at(15));
    builder.terminate(ret(joined), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    assert!(body.block(join).params().is_empty());
    assert_eq!(edge_of(body.block(join).insts().nth(0).expect("the block holds that instruction")).before, body.order_param().expect("ordered"));
}

#[test]
fn the_token_crosses_a_loop_back_edge() {
    let mut f = fixture();
    let (nc, int, bool, whole) = (f.name("c"), f.int, f.bool, f.whole);
    let body_call = f.call(whole);
    let mut builder = build(&mut f.module, true, f.bool);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header), acvus_ast::Span::ZERO);
    builder.switch_to(header);
    builder.terminate(Terminator::While {
        cond: c,
        body: loop_body,
        body_args: Vec::new(),
        exit,
        exit_args: Vec::new(),
    }, acvus_ast::Span::ZERO);
    builder.switch_to(loop_body);
    builder.push_effect(body_call, int, at(16));
    builder.terminate(jump(header), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    builder.terminate(ret(c), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    let [token] = body.block(header).params() else {
        panic!("the header takes the token around the loop: {:?}", body.block(header).params())
    };
    let body_edge = edge_of(body.block(loop_body).insts().nth(0).expect("the block holds that instruction"));
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
    let mut builder = build(&mut f.module, true, f.int);
    builder.open_anyorder();
    builder.push_effect(first, int, at(17));
    builder.open_anyorder();
    builder.push_effect(second, int, at(18));
    builder.close_anyorder();
    builder.close_anyorder();
    let last = builder.push_effect(after, int, at(19));
    builder.terminate(ret(last), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
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
    // A scope's merge is placed with the span of the call it follows.
    assert_eq!((insts[1].span(), insts[3].span()), (at(17), at(18)));
    assert_eq!(edge_of(&insts[4]).before, m2);
}

#[test]
fn anyorder_merges_on_an_edge_that_leaves_it() {
    let mut f = fixture();
    let (int, whole) = (f.int, f.whole);
    let (inside, outside) = (f.call(whole), f.call(whole));
    let mut builder = build(&mut f.module, true, f.int);
    let out = builder.reserve_block();
    builder.open_anyorder();
    builder.push_effect(inside, int, at(20));
    builder.terminate(jump(out), acvus_ast::Span::ZERO);
    let rest = builder.new_block(&[]);
    builder.switch_to(rest);
    builder.close_anyorder();
    builder.terminate(Terminator::Diverge, acvus_ast::Span::ZERO);
    builder.make_block(out, &[]);
    builder.switch_to(out);
    let last = builder.push_effect(outside, int, at(21));
    builder.terminate(ret(last), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
    let merge = result_of(&insts[1]);
    assert!(matches!(insts[1].op(), Op::Merge { .. }));
    assert_eq!(edge_of(body.block(out).insts().nth(0).expect("the block holds that instruction")).before, merge);
}

#[test]
fn a_return_inside_anyorder_carries_the_merge() {
    let mut f = fixture();
    let (int, whole) = (f.int, f.whole);
    let inside = f.call(whole);
    let mut builder = build(&mut f.module, true, f.int);
    builder.open_anyorder();
    let value = builder.push_effect(inside, int, at(22));
    builder.terminate(ret(value), acvus_ast::Span::ZERO);
    let rest = builder.new_block(&[]);
    builder.switch_to(rest);
    builder.close_anyorder();
    builder.terminate(Terminator::Diverge, acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let merge = result_of(body.block(body.entry()).insts().nth(1).expect("the block holds that instruction"));
    assert_eq!(returned_order(body, body.entry()), Some(merge));
}

#[test]
fn removing_an_effectful_call_rethreads_the_token_around_it() {
    let mut f = fixture();
    let (int, whole) = (f.int, f.whole);
    let (first_call, second_call) = (f.call(whole), f.call(whole));
    let mut builder = build(&mut f.module, true, f.int);
    let first = builder.push_effect(first_call, int, at(23));
    let second = builder.push_effect(second_call, int, at(24));
    builder.terminate(ret(second), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let o0 = f.body(id).order_param().expect("ordered");

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(first), Ok(()));
    let body = rewriter.body();
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
    assert_eq!(insts.len(), 1);
    assert_eq!(result_of(&insts[0]), second);
    assert_eq!(edge_of(&insts[0]).before, o0);
    assert_eq!(body.use_count(o0), 1);
    assert!(rewriter.finish().touched(Aspect::OrderTokens));
}

/// A `Take` of the whole of `context`.
fn take_context(context: QualifiedRef) -> Op {
    Op::Take { target: RefTarget::Context(context), path: Vec::new(), taken_out: false }
}

/// context §3: a volatile context's take and assign are effects, so each
/// stands on the chain as a call whose effect is not pure does.
#[test]
fn a_volatile_context_take_and_assign_are_threaded_on_the_chain() {
    let mut f = fixture();
    let (int, page) = (f.int, QualifiedRef::root(f.name("page")));
    let mut builder = build(&mut f.module, true, f.int);
    builder.context(page, int, true);
    let taken = builder.push_effect(take_context(page), int, at(30));
    let assign = Op::Assign { target: RefTarget::Context(page), path: Vec::new(), value: taken, restores: false };
    builder.push_effect_unit(assign, at(31));
    builder.terminate(ret(taken), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let o0 = body.order_param().expect("an ordered body takes an order token");
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
    let take = edge_of(&insts[0]);
    let store = edge_of(&insts[1]);
    assert_eq!(take.before, o0);
    assert_eq!(store.before, take.after);
    assert_eq!(insts[1].result(), None);
    assert_eq!(returned_order(body, body.entry()), Some(store.after));
}

#[test]
#[should_panic(expected = "issues no effect")]
fn a_pure_body_takes_no_context_on_the_chain() {
    let mut f = fixture();
    let (int, page) = (f.int, QualifiedRef::root(f.name("page")));
    let mut builder = build(&mut f.module, false, f.int);
    builder.context(page, int, true);
    builder.push_effect(take_context(page), int, at(34));
}

#[test]
#[should_panic(expected = "which the body does not name")]
fn a_context_the_body_does_not_name_is_not_taken() {
    let mut f = fixture();
    let (int, page) = (f.int, QualifiedRef::root(f.name("page")));
    build(&mut f.module, false, f.int).push(take_context(page), int, at(32));
}

#[test]
fn removing_a_take_on_the_chain_rethreads_the_token_around_it() {
    let mut f = fixture();
    let (int, page) = (f.int, QualifiedRef::root(f.name("page")));
    let mut builder = build(&mut f.module, true, f.int);
    builder.context(page, int, true);
    let unread = builder.push_effect(take_context(page), int, at(35));
    let read = builder.push_effect(take_context(page), int, at(36));
    builder.terminate(ret(read), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let o0 = f.body(id).order_param().expect("ordered");

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(unread), Ok(()));
    let body = rewriter.body();
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
    assert_eq!(insts.len(), 1);
    assert_eq!(edge_of(&insts[0]).before, o0);
    assert!(rewriter.finish().touched(Aspect::OrderTokens));
}

// -- Interning through the Builder ---------------------------------------

#[test]
fn intern_finds_a_type_already_in_the_table_and_adds_a_new_one() {
    let mut f = fixture();
    let bool = f.bool;
    let mut builder = build(&mut f.module, false, f.int);
    assert_eq!(builder.intern(TyKind::Bool), bool);
    let u64 = builder.intern(TyKind::Int(IntTy::U64));
    builder.seal().expect_err("the entry block has no terminator");
    assert_eq!(f.module.types().kind(u64), &TyKind::Int(IntTy::U64));
}

#[test]
#[should_panic(expected = "types §4")]
fn intern_refuses_a_reference_to_a_reference() {
    let mut f = fixture();
    let int = f.int;
    let mut builder = build(&mut f.module, false, f.int);
    let inner = builder.intern(TyKind::Ref(Mutability::Shared, int));
    builder.intern(TyKind::Ref(Mutability::Shared, inner));
}

// -- 5. Spawn and TakeEffect are made only together (effects §6) ---------

#[test]
fn split_call_makes_a_start_and_its_taking() {
    let mut f = fixture();
    let (int, splittable) = (f.int, f.splittable);
    let call_op = f.call(splittable);
    let mut builder = build(&mut f.module, true, f.int);
    let call = builder.push_effect(call_op, int, at(25));
    let user = builder.push(neg(call), int, at(26));
    builder.terminate(ret(user), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let entry = f.body(id).entry();
    let call_edge = edge_of(f.body(id).block(entry).insts().nth(0).expect("the block holds that instruction"));

    let mut rewriter = f.module.rewrite(id);
    rewriter.split_call(call);
    let body = rewriter.body();
    let insts: Vec<_> = body.block(entry).insts().collect();
    let handle = result_of(&insts[0]);
    assert!(matches!(insts[0].op(), Op::Spawn { .. }));
    assert_eq!(insts[0].order(), Some(OrderUse::Start(call_edge.before)));
    assert!(matches!(insts[1].op(), Op::TakeEffect { src } if *src == handle));
    assert_eq!(insts[1].result(), Some(call));
    assert_eq!(insts[1].order(), Some(OrderUse::Take(call_edge.after)));
    assert!(matches!(insts[2].op(), Op::UnaryOp { operand, .. } if *operand == call));
    assert!(matches!(rewriter.types().kind(body.ty_of(handle)), TyKind::Handle(ret) if *ret == f.int));
    assert_eq!(body.use_count(handle), 1);
    assert_eq!((insts[0].span(), insts[1].span()), (at(25), at(25)));
    let changes = rewriter.finish();
    assert!(changes.touched(Aspect::Instructions));
    assert!(changes.touched(Aspect::OrderTokens));
}

#[test]
fn removing_a_taking_removes_its_start_and_rethreads_the_token() {
    let mut f = fixture();
    let (int, splittable, whole) = (f.int, f.splittable, f.whole);
    let (call_op, later_op) = (f.call(splittable), f.call(whole));
    let mut builder = build(&mut f.module, true, f.int);
    let call = builder.push_effect(call_op, int, at(27));
    let user = builder.push(neg(call), int, at(28));
    let later = builder.push_effect(later_op, int, at(29));
    builder.terminate(ret(later), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let o0 = f.body(id).order_param().expect("ordered");

    let mut rewriter = f.module.rewrite(id);
    rewriter.split_call(call);
    assert_eq!(rewriter.remove(call), Err(StillUsed { value: call }));
    assert_eq!(rewriter.remove(user), Ok(()));
    assert_eq!(rewriter.remove(call), Ok(()));
    let body = rewriter.body();
    let insts: Vec<_> = body.block(body.entry()).insts().collect();
    assert_eq!(insts.len(), 1);
    assert_eq!(result_of(&insts[0]), later);
    assert_eq!(edge_of(&insts[0]).before, o0);
}

#[test]
#[should_panic(expected = "may be split")]
fn a_call_its_callee_does_not_let_split_is_refused() {
    let mut f = fixture();
    let (int, call_op) = (f.int, f.call(f.whole));
    let mut builder = build(&mut f.module, true, f.int);
    let call = builder.push_effect(call_op, int, at(30));
    builder.terminate(ret(call), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    f.module.rewrite(id).split_call(call);
}

#[test]
#[should_panic(expected = "written only by the core")]
fn a_start_is_not_pushed_alone() {
    let mut f = fixture();
    let (int, splittable, name) = (f.int, f.splittable, f.name("f"));
    build(&mut f.module, false, f.int).push(
        Op::Spawn { callee: Callee::Direct(QualifiedRef::root(name)), callee_ty: splittable, args: Vec::new() },
        int,
        at(31),
    );
}

// -- 6. A Diamond's join and a loop's body and exit follow every edit ---

#[test]
fn a_retarget_updates_a_diamonds_join() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let [then_block, else_block, join, new_join] = [(); 4].map(|()| builder.new_block(&[]));
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(jump(join), acvus_ast::Span::ZERO);
    }
    for block in [join, new_join] {
        builder.switch_to(block);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
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
fn a_retarget_updates_a_loops_body_and_exit() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let header = builder.current_block();
    let [loop_body, exit, other_body, other_exit] = [(); 4].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::While { cond: c, body: loop_body, body_args: Vec::new(), exit, exit_args: Vec::new() }, acvus_ast::Span::ZERO);
    for block in [loop_body, other_body] {
        builder.switch_to(block);
        builder.terminate(jump(header), acvus_ast::Span::ZERO);
    }
    for block in [exit, other_exit] {
        builder.switch_to(block);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
    }
    let id = builder.seal().expect("the body has no open state");

    let mut rewriter = f.module.rewrite(id);
    rewriter.retarget(loop_body, other_body);
    rewriter.retarget(exit, other_exit);
    let Terminator::While { body, exit: now, .. } = rewriter.body().block(header).terminator() else {
        panic!("the header still ends in its loop")
    };
    assert_eq!(*body, other_body);
    assert_eq!(*now, other_exit);
}

#[test]
fn a_split_keeps_the_block_named_and_moves_the_diamond_with_the_tail() {
    let mut f = fixture();
    let (na, nc, nd, int, bool) = (f.name("a"), f.name("c"), f.name("d"), f.int, f.bool);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let d = builder.param(nd, bool, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    let x = builder.push(add(a, a), int, at(32));
    let y = builder.push(neg(x), int, at(33));
    builder.terminate(diamond(c, then_block, else_block, join), acvus_ast::Span::ZERO);
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(jump(join), acvus_ast::Span::ZERO);
    }
    builder.switch_to(join);
    builder.terminate(ret(y), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let second = f.body(id).inst_at(entry, 1);
    let mut rewriter = f.module.rewrite(id);
    let tail = rewriter.split_block(entry, second);
    let body = rewriter.body();
    assert!(matches!(body.block(entry).terminator(), Terminator::Jump { to, args } if *to == tail && args.is_empty()));
    assert!(matches!(body.block(tail).terminator(), Terminator::Diamond { join: j, .. } if *j == join));
    assert_eq!(body.block(tail).insts().len(), 1);
    // The moved instruction's definition and uses moved with it.
    rewriter.replace_all_uses(x, a);
    assert_eq!(rewriter.remove(x), Ok(()));
    let moved = rewriter.body().block(tail).insts().nth(0).expect("the block holds that instruction");
    assert_eq!(moved.result(), Some(y));
    assert!(matches!(moved.op(), Op::UnaryOp { operand, .. } if *operand == a));
    // The Diamond's use of `c` moved with it to the tail's terminator.
    rewriter.replace_all_uses(c, d);
    assert!(matches!(rewriter.body().block(tail).terminator(), Terminator::Diamond { cond, .. } if *cond == d));
    assert_eq!(rewriter.body().use_count(c), 0);
}

/// `switch a { 1 => one, 2 => two, _ => other }` joining at `join`, and a
/// block `spare` that also returns. Every arm jumps to the join.
struct SwitchBody {
    id: BodyId,
    entry: BlockId,
    arms: [BlockId; 3],
    join: BlockId,
    spare: BlockId,
}

fn switch_body(f: &mut Fixture) -> SwitchBody {
    let (na, int) = (f.name("a"), f.int);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let entry = builder.current_block();
    let [one, two, other, join, spare] = [(); 5].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::Switch {
        tag: a,
        arms: vec![(SwitchKey::Int(1), one, Vec::new()), (SwitchKey::Int(2), two, Vec::new())],
        default: Some((other, Vec::new())),
        join,
    }, acvus_ast::Span::ZERO);
    for block in [one, two, other] {
        builder.switch_to(block);
        builder.terminate(jump(join), acvus_ast::Span::ZERO);
    }
    for block in [join, spare] {
        builder.switch_to(block);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
    }
    let id = builder.seal().expect("the body has no open state");
    SwitchBody { id, entry, arms: [one, two, other], join, spare }
}

fn switch_join(body: &Body, block: BlockId) -> BlockId {
    let Terminator::Switch { join, .. } = body.block(block).terminator() else {
        panic!("{block:?} ends in its Switch")
    };
    *join
}

#[test]
fn a_retarget_updates_a_switchs_join() {
    let mut f = fixture();
    let s = switch_body(&mut f);
    let mut rewriter = f.module.rewrite(s.id);
    rewriter.retarget(s.join, s.spare);
    let body = rewriter.body();
    assert_eq!(switch_join(body, s.entry), s.spare);
    for arm in s.arms {
        assert!(matches!(body.block(arm).terminator(), Terminator::Jump { to, .. } if *to == s.spare));
    }
}

#[test]
fn a_split_keeps_a_switchs_join_named_and_moves_the_switch_with_the_tail() {
    let mut f = fixture();
    let s = switch_body(&mut f);
    let (join_first, entry_first) = (f.body(s.id).first_inst(s.join), f.body(s.id).first_inst(s.entry));
    let mut rewriter = f.module.rewrite(s.id);
    let join_tail = rewriter.split_block(s.join, join_first);
    let entry_tail = rewriter.split_block(s.entry, entry_first);
    let body = rewriter.body();
    assert_eq!(switch_join(body, entry_tail), s.join, "the join keeps its id, and the Switch still names it");
    assert!(matches!(body.block(s.join).terminator(), Terminator::Jump { to, .. } if *to == join_tail));
    assert!(matches!(body.block(s.entry).terminator(), Terminator::Jump { to, .. } if *to == entry_tail));
}

#[test]
#[should_panic(expected = "is not a block of this body")]
fn the_builder_refuses_a_switch_whose_join_is_not_a_block() {
    let mut f = fixture();
    let (na, int) = (f.name("a"), f.int);
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let arm = builder.new_block(&[]);
    builder.terminate(Terminator::Switch {
        tag: a,
        arms: vec![(SwitchKey::Int(1), arm, Vec::new())],
        default: None,
        join: BlockId::of_index(7),
    }, acvus_ast::Span::ZERO);
}

// -- Builder: a nested body --------------------------------------------------

/// A lambda's body is built while its enclosing body is open, and the
/// enclosing body's `MakeClosure` names the id the nested seal gave.
#[test]
fn a_nested_body_is_built_while_the_enclosing_one_is_open() {
    let mut f = fixture();
    let (na, nx, nk, int) = (f.name("a"), f.name("x"), f.name("k"), f.int);
    let lambda_ty = f.module.types.intern(TyKind::Fn {
        params: Box::new([int]),
        ret: int,
        effect: EffectTerm::Known(Effect::pure()),
        flows: FlowTerm::Known(Flows::none()),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    let mut outer = build(&mut f.module, false, int);
    let a = outer.param(na, int, acvus_ast::Span::ZERO);
    let k = outer.push(Op::Const { value: Literal::Int(1) }, int, at(1));
    let lambda = {
        let mut inner = outer.build_nested(false, int);
        let x = inner.param(nx, int, acvus_ast::Span::ZERO);
        let captured = inner.capture(nk, int, acvus_ast::Span::ZERO);
        let sum = inner.push(add(x, captured), int, at(2));
        inner.terminate(ret(sum), acvus_ast::Span::ZERO);
        inner.seal().expect("the lambda's body has no open state")
    };
    let closure = outer.push(Op::MakeClosure { bodies: vec![lambda], captures: vec![k] }, lambda_ty, at(3));
    let called = outer.push(
        Op::FunctionCall { callee: Callee::Indirect(closure), callee_ty: lambda_ty, args: vec![a] },
        int,
        at(4),
    );
    outer.terminate(ret(called), acvus_ast::Span::ZERO);
    let enclosing = outer.seal().expect("the enclosing body has no open state");

    assert_ne!(lambda, enclosing);
    assert_eq!(f.body(lambda).params().len(), 1);
    assert_eq!(f.body(lambda).captures().len(), 1);
    let entry = f.body(enclosing).entry();
    assert!(matches!(
        f.body(enclosing).block(entry).insts().nth(1).expect("the block holds that instruction").op(),
        Op::MakeClosure { bodies, .. } if bodies[..] == [lambda]
    ));
}

// -- Builder: seal reports each kind of open state -----------------------

#[test]
fn seal_reports_a_block_without_its_terminator() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, false, f.int);
    let open = builder.new_block(&[]);
    builder.terminate(Terminator::Diverge, acvus_ast::Span::ZERO);
    assert_eq!(builder.seal().err(), Some(Unsealed::BlockWithoutTerminator(open)));
}

#[test]
fn seal_reports_an_edge_to_an_unmade_block() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, false, f.int);
    let unmade = builder.reserve_block();
    builder.terminate(jump(unmade), acvus_ast::Span::ZERO);
    assert_eq!(builder.seal().err(), Some(Unsealed::EdgeToUnmadeBlock(unmade)));
}

#[test]
fn seal_reports_a_reserved_block_nothing_names() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, false, f.int);
    let unmade = builder.reserve_block();
    builder.terminate(Terminator::Diverge, acvus_ast::Span::ZERO);
    assert_eq!(builder.seal().err(), Some(Unsealed::ReservedNeverMade(unmade)));
}

#[test]
fn seal_reports_an_open_anyorder_scope() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, true, f.int);
    builder.open_anyorder();
    builder.terminate(Terminator::Diverge, acvus_ast::Span::ZERO);
    assert_eq!(builder.seal().err(), Some(Unsealed::OpenAnyorder));
}

#[test]
fn a_sealed_body_is_added_to_the_module() {
    let mut f = fixture();
    let mut builder = build(&mut f.module, false, f.int);
    builder.open_anyorder();
    builder.close_anyorder();
    builder.terminate(Terminator::Diverge, acvus_ast::Span::ZERO);
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
    let mut builder = build(&mut f.module, false, f.int);
    let a = builder.param(na, int, acvus_ast::Span::ZERO);
    let flag = builder.push(Op::Const { value: Literal::Bool(false) }, bool, at(34));
    builder.terminate(Terminator::Return { value: a, order: Some(flag), span: Span::ZERO }, Span::ZERO);
}

// -- Order tokens are the core's alone (ir-core-structure §2) -------------

#[test]
fn the_core_records_every_token_it_makes() {
    let mut f = fixture();
    let (nc, int, bool, whole) = (f.name("c"), f.int, f.bool, f.whole);
    let (loop_call, scoped_call) = (f.call(whole), f.call(whole));
    let mut builder = build(&mut f.module, true, f.int);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header), acvus_ast::Span::ZERO);
    builder.switch_to(header);
    builder.terminate(Terminator::While { cond: c, body: loop_body, body_args: Vec::new(), exit, exit_args: Vec::new() }, acvus_ast::Span::ZERO);
    builder.switch_to(loop_body);
    let looped = builder.push_effect(loop_call, int, at(40));
    builder.terminate(jump(header), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    builder.open_anyorder();
    let scoped = builder.push_effect(scoped_call, int, at(41));
    builder.close_anyorder();
    builder.terminate(ret(scoped), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let body = f.body(id);
    let order_param = body.order_param().expect("an ordered body takes an order token");
    let yielded = edge_of(body.block(loop_body).insts().nth(0).expect("the block holds that instruction")).after;
    let [placed] = body.block(header).params() else {
        panic!("the header takes the token around the loop: {:?}", body.block(header).params())
    };
    let merge = body.block(exit).insts().find(|inst| matches!(inst.op(), Op::Merge { .. }));
    let merged = result_of(merge.expect("the anyorder scope merges what its call yielded"));
    for token in [order_param, yielded, *placed, merged] {
        assert!(body.is_token(token), "{token:?} is a token the core made");
    }
    for value in [c, looped, scoped] {
        assert!(!body.is_token(value), "{value:?} is a value");
    }
}

#[test]
#[should_panic(expected = "only the core makes")]
fn a_parameter_of_order_type_is_refused() {
    let mut f = fixture();
    let (name, order) = (f.name("o"), f.order);
    build(&mut f.module, false, f.int).param(name, order, acvus_ast::Span::ZERO);
}

#[test]
#[should_panic(expected = "only the core makes")]
fn a_capture_of_order_type_is_refused() {
    let mut f = fixture();
    let (name, order) = (f.name("o"), f.order);
    build(&mut f.module, false, f.int).capture(name, order, acvus_ast::Span::ZERO);
}

#[test]
#[should_panic(expected = "only the core makes")]
fn a_slot_of_order_type_is_refused() {
    let mut f = fixture();
    let order = f.order;
    build(&mut f.module, false, f.int).slot(order, at(0));
}

#[test]
#[should_panic(expected = "only the core makes")]
fn a_block_parameter_of_order_type_is_refused() {
    let mut f = fixture();
    let order = f.order;
    build(&mut f.module, true, f.int).new_block(&[order]);
}

#[test]
#[should_panic(expected = "only the core makes")]
fn a_result_of_order_type_is_refused() {
    let mut f = fixture();
    let order = f.order;
    build(&mut f.module, true, f.int).push(Op::Undef, order, at(42));
}
