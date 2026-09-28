//! The core's invariants (core.rs), each test citing the one it checks.
//! Bodies are built through the Builder; instructions are placed through
//! the core itself until the Builder's `push` is settled.

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, QualifiedRef};
use smallvec::SmallVec;

use super::body::Def;
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
    splittable: Ty,
    whole: Ty,
}

fn fixture() -> Fixture {
    let mut types = Types::<Concrete>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let bool = types.intern(TyKind::Bool);
    let order = types.intern(TyKind::Order);
    let mut call_ty = |split: SplitCost| {
        let effect = Effect::new(Reissue::Opaque, false, Contexts::new(), Contexts::new(), split);
        types.intern(TyKind::Fn {
            params: Box::new([]),
            ret: int,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        })
    };
    let splittable = call_ty(SplitCost::WithCost(Cost::stated(1)));
    let whole = call_ty(SplitCost::Unavailable);
    Fixture {
        module: Module::new(types, int, Flows::none()),
        interner: Interner::new(),
        int,
        bool,
        order,
        splittable,
        whole,
    }
}

impl Fixture {
    fn name(&self, name: &str) -> acvus_utils::Astr {
        self.interner.intern(name)
    }

    fn callee(&self) -> QualifiedRef {
        QualifiedRef::root(self.name("f"))
    }

    /// Appends at the end of `block` the instruction `build` makes from new
    /// values of `result_tys`.
    fn append(
        &mut self,
        body: BodyId,
        block: BlockId,
        result_tys: &[Ty],
        build: impl FnOnce(&[ValueId]) -> Op,
    ) -> SmallVec<[ValueId; 2]> {
        let Module { bodies, types, .. } = &mut self.module;
        let body = &mut bodies[body.0 as usize];
        let end = body.blocks[block.index()].insts.len();
        body.insert_new_inst(types, block, end, result_tys, build, Span::ZERO)
    }

    fn body(&self, body: BodyId) -> &Body {
        self.module.body(body)
    }

    /// Gives a sealed body an order parameter, which only lower's order
    /// threading will declare once it is settled.
    fn give_order_param(&mut self, body: BodyId) -> ValueId {
        let Module { bodies, types, .. } = &mut self.module;
        let body = &mut bodies[body.0 as usize];
        let token = body.make_value(types, self.order, Def::OrderParam);
        body.order_param = Some(token);
        token
    }
}

fn ret(value: ValueId) -> Terminator {
    Terminator::Return { value, order: None, span: Span::ZERO }
}

fn binop(dst: ValueId, left: ValueId, right: ValueId) -> Op {
    Op::BinOp { dst, op: BinOp::Add(Overflow::Trap), left, right }
}

fn neg(dst: ValueId, operand: ValueId) -> Op {
    Op::UnaryOp { dst, op: UnaryOp::Neg(Overflow::Trap), operand }
}

/// One block that returns its one `i64` parameter.
fn single_block(f: &mut Fixture) -> (BodyId, ValueId) {
    let name = f.name("a");
    let int = f.int;
    let mut builder = f.module.build_body();
    let a = builder.param(name, int);
    builder.terminate(ret(a));
    (builder.seal().expect("the body has no open state"), a)
}

// -- 1. A value is made with its type, in the module's one table ---------

#[test]
fn every_value_is_born_typed() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = f.module.build_body();
    let a = builder.param(name, int);
    let join = builder.new_block(&[bool]);
    let p = builder.block_params(join)[0];
    builder.terminate(Terminator::Diverge);
    builder.switch_to(join);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let body = f.body(id);
    assert_eq!(body.ty_of(a), int);
    assert_eq!(body.ty_of(p), bool);
}

#[test]
#[should_panic(expected = "types §4")]
fn a_reference_to_a_reference_is_refused() {
    let mut f = fixture();
    let name = f.name("r");
    let int = f.int;
    let Module { types, .. } = &mut f.module;
    let inner = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    let outer = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(inner)));
    f.module.build_body().param(name, outer);
}

#[test]
#[should_panic(expected = "read in table")]
fn a_type_of_another_table_is_refused() {
    let mut f = fixture();
    let name = f.name("a");
    let foreign = Types::<Concrete>::new().intern(TyKind::Bool);
    f.module.build_body().param(name, foreign);
}

// -- 2. Def-use; a used value is removed only with its replacement -------

#[test]
fn removing_a_used_value_is_refused() {
    let mut f = fixture();
    let (id, a) = single_block(&mut f);
    let entry = f.body(id).entry();
    let x = f.append(id, entry, &[f.int], |v| binop(v[0], a, a))[0];
    f.append(id, entry, &[f.int], |v| neg(v[0], x));

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(x), Err(StillUsed { value: x }));
    assert_eq!(rewriter.body().use_count(x), 1);
    assert_eq!(rewriter.body().use_count(a), 3);
    assert_eq!(rewriter.body().block(entry).insts().len(), 2);
    assert!(!rewriter.finish().touched(Aspect::Instructions));
}

#[test]
fn a_replaced_value_is_removed_and_its_operands_lose_their_uses() {
    let mut f = fixture();
    let (id, a) = single_block(&mut f);
    let entry = f.body(id).entry();
    let x = f.append(id, entry, &[f.int], |v| binop(v[0], a, a))[0];
    let y = f.append(id, entry, &[f.int], |v| neg(v[0], x))[0];

    let mut rewriter = f.module.rewrite(id);
    rewriter.replace_all_uses(x, a);
    assert_eq!(rewriter.body().use_count(a), 4);
    assert_eq!(rewriter.remove(x), Ok(()));
    let body = rewriter.body();
    assert_eq!(body.use_count(a), 2);
    let insts = body.block(entry).insts();
    assert_eq!(insts.len(), 1);
    assert!(matches!(insts[0].0, Op::UnaryOp { dst, operand, .. } if dst == y && operand == a));
    let changes = rewriter.finish();
    assert!(changes.touched(Aspect::Instructions));
    assert!(!changes.touched(Aspect::CfgShape));
}

#[test]
fn a_removal_moves_the_definitions_after_it() {
    let mut f = fixture();
    let (id, a) = single_block(&mut f);
    let entry = f.body(id).entry();
    let x = f.append(id, entry, &[f.int], |v| binop(v[0], a, a))[0];
    let y = f.append(id, entry, &[f.int], |v| neg(v[0], a))[0];
    let z = f.append(id, entry, &[f.int], |v| neg(v[0], y))[0];

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(x), Ok(()));
    // `y` now stands first: replacing it must find its use in `z`, which
    // moved up with it.
    rewriter.replace_all_uses(y, a);
    assert_eq!(rewriter.remove(y), Ok(()));
    let insts = rewriter.body().block(entry).insts();
    assert!(matches!(insts[0].0, Op::UnaryOp { dst, operand, .. } if dst == z && operand == a));
}

#[test]
#[should_panic(expected = "replaced only by a value of its type")]
fn a_use_is_replaced_only_by_a_value_of_its_type() {
    let mut f = fixture();
    let (id, a) = single_block(&mut f);
    let entry = f.body(id).entry();
    let t = f.append(id, entry, &[f.bool], |v| Op::Const { dst: v[0], value: Literal::Bool(true) })[0];
    f.module.rewrite(id).replace_all_uses(a, t);
}

// -- 3. Block parameters agree with every edge's arguments ---------------

#[test]
#[should_panic(expected = "passes 0 arguments to 1 parameters")]
fn an_edge_short_of_arguments_is_refused() {
    let mut f = fixture();
    let int = f.int;
    let mut builder = f.module.build_body();
    let join = builder.new_block(&[int]);
    builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
}

#[test]
#[should_panic(expected = "passes")]
fn an_edge_argument_of_another_type_is_refused() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = f.module.build_body();
    let a = builder.param(name, int);
    let join = builder.new_block(&[bool]);
    builder.terminate(Terminator::Jump { to: join, args: vec![a] });
}

#[test]
#[should_panic(expected = "passes")]
fn a_reserved_block_is_checked_against_its_edges_when_made() {
    let mut f = fixture();
    let (name, int, bool) = (f.name("a"), f.int, f.bool);
    let mut builder = f.module.build_body();
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
    let mut builder = f.module.build_body();
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
    let mut builder = f.module.build_body();
    let a = builder.param(na, int);
    let b = builder.param(nb, int);
    let c = builder.param(nc, bool);
    let then_block = builder.new_block(&[]);
    let else_block = builder.new_block(&[]);
    let join = builder.reserve_block();
    builder.terminate(Terminator::Diamond {
        cond: c,
        then_to: then_block,
        then_args: Vec::new(),
        else_to: else_block,
        else_args: Vec::new(),
        join,
    });
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

// -- 4. The order token chain --------------------------------------------

#[test]
fn removing_an_effectful_call_rethreads_the_token_around_it() {
    let mut f = fixture();
    let (id, _) = single_block(&mut f);
    let entry = f.body(id).entry();
    let o0 = f.give_order_param(id);
    let (int, order, whole, callee) = (f.int, f.order, f.whole, f.callee());
    let first = f.append(id, entry, &[int, order], |v| {
        call_op(callee, whole, v[0], Some(OrderEdge { before: o0, after: v[1] }))
    });
    let (first_call, o1) = (first[0], first[1]);
    let second = f.append(id, entry, &[int, order], |v| {
        call_op(callee, whole, v[0], Some(OrderEdge { before: o1, after: v[1] }))
    });

    let mut rewriter = f.module.rewrite(id);
    assert_eq!(rewriter.remove(first_call), Ok(()));
    let body = rewriter.body();
    let insts = body.block(entry).insts();
    assert_eq!(insts.len(), 1);
    assert!(matches!(
        insts[0].0,
        Op::FunctionCall { dst, order: Some(OrderEdge { before, after }), .. }
            if dst == second[0] && before == o0 && after == second[1]
    ));
    assert_eq!(body.use_count(o0), 1);
    let changes = rewriter.finish();
    assert!(changes.touched(Aspect::OrderTokens));
}

// -- 5. Spawn and TakeEffect are made only together ----------------------

fn split_fixture(f: &mut Fixture) -> (BodyId, BlockId, ValueId, ValueId, ValueId, ValueId) {
    let (id, _) = single_block(f);
    let entry = f.body(id).entry();
    let o0 = f.give_order_param(id);
    let (int, order, splittable, callee) = (f.int, f.order, f.splittable, f.callee());
    let made = f.append(id, entry, &[int, order], |v| {
        call_op(callee, splittable, v[0], Some(OrderEdge { before: o0, after: v[1] }))
    });
    let (call, o1) = (made[0], made[1]);
    let user = f.append(id, entry, &[int], |v| neg(v[0], call))[0];
    (id, entry, o0, call, o1, user)
}

fn call_op(callee: QualifiedRef, callee_ty: Ty, dst: ValueId, order: Option<OrderEdge>) -> Op {
    Op::FunctionCall {
        dst,
        callee: Callee::Direct(callee),
        callee_ty,
        args: Vec::new(),
        order,
    }
}

#[test]
fn split_call_makes_a_start_and_its_taking() {
    let mut f = fixture();
    let (id, entry, o0, call, o1, user) = split_fixture(&mut f);

    let mut rewriter = f.module.rewrite(id);
    rewriter.split_call(call);
    let body = rewriter.body();
    let insts = body.block(entry).insts();
    let Op::Spawn { dst: handle, order: Some(before), .. } = insts[0].0 else {
        panic!("the call's place holds its start: {:?}", insts[0].0)
    };
    assert_eq!(before, o0);
    assert!(matches!(
        insts[1].0,
        Op::TakeEffect { dst, src, order: Some(after) } if dst == call && src == handle && after == o1
    ));
    assert!(matches!(insts[2].0, Op::UnaryOp { dst, operand, .. } if dst == user && operand == call));
    assert!(matches!(rewriter.types().kind(body.ty_of(handle)), TyKind::Handle(ret) if *ret == f.int));
    assert_eq!(body.use_count(handle), 1);
    assert_eq!(body.use_count(o0), 1);
    let changes = rewriter.finish();
    assert!(changes.touched(Aspect::Instructions));
    assert!(changes.touched(Aspect::OrderTokens));
}

#[test]
fn removing_a_taking_removes_its_start_and_rethreads_the_token() {
    let mut f = fixture();
    let (id, entry, o0, call, o1, user) = split_fixture(&mut f);
    let (int, order, whole, callee) = (f.int, f.order, f.whole, f.callee());
    let later = f.append(id, entry, &[int, order], |v| {
        call_op(callee, whole, v[0], Some(OrderEdge { before: o1, after: v[1] }))
    });

    let mut rewriter = f.module.rewrite(id);
    rewriter.split_call(call);
    assert_eq!(rewriter.remove(call), Err(StillUsed { value: call }));
    assert_eq!(rewriter.remove(user), Ok(()));
    assert_eq!(rewriter.remove(call), Ok(()));
    let insts = rewriter.body().block(entry).insts();
    assert_eq!(insts.len(), 1);
    assert!(matches!(
        insts[0].0,
        Op::FunctionCall { dst, order: Some(OrderEdge { before, .. }), .. } if dst == later[0] && before == o0
    ));
}

#[test]
#[should_panic(expected = "may be split")]
fn a_call_its_callee_does_not_let_split_is_refused() {
    let mut f = fixture();
    let (id, _) = single_block(&mut f);
    let entry = f.body(id).entry();
    let (int, whole, callee) = (f.int, f.whole, f.callee());
    let whole_call = f.append(id, entry, &[int], |v| call_op(callee, whole, v[0], None))[0];
    f.module.rewrite(id).split_call(whole_call);
}

// -- 6. A Diamond's join and a loop's stages and exit follow every edit ---

#[test]
fn a_retarget_updates_a_diamonds_join() {
    let mut f = fixture();
    let (na, nc, int, bool) = (f.name("a"), f.name("c"), f.int, f.bool);
    let mut builder = f.module.build_body();
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let then_block = builder.new_block(&[]);
    let else_block = builder.new_block(&[]);
    let join = builder.new_block(&[]);
    let new_join = builder.new_block(&[]);
    builder.terminate(Terminator::Diamond {
        cond: c,
        then_to: then_block,
        then_args: Vec::new(),
        else_to: else_block,
        else_args: Vec::new(),
        join,
    });
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
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
    let mut builder = f.module.build_body();
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let [loop_body, exit, other_body, other_exit] = [(); 4].map(|()| builder.new_block(&[]));
    let header = builder.current_block();
    builder.terminate(Terminator::While {
        cond: c,
        stages: Stages::lowered(loop_body),
        exit,
        exit_args: Vec::new(),
    });
    for block in [loop_body, other_body] {
        builder.switch_to(block);
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() });
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
    let mut builder = f.module.build_body();
    let a = builder.param(na, int);
    let c = builder.param(nc, bool);
    let d = builder.param(nd, bool);
    let entry = builder.current_block();
    let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::Diamond {
        cond: c,
        then_to: then_block,
        then_args: Vec::new(),
        else_to: else_block,
        else_args: Vec::new(),
        join,
    });
    for block in [then_block, else_block] {
        builder.switch_to(block);
        builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
    }
    builder.switch_to(join);
    builder.terminate(ret(a));
    let id = builder.seal().expect("the body has no open state");
    let x = f.append(id, entry, &[int], |v| binop(v[0], a, a))[0];
    let y = f.append(id, entry, &[int], |v| neg(v[0], x))[0];

    let mut rewriter = f.module.rewrite(id);
    let tail = rewriter.split_block(entry, 1);
    let body = rewriter.body();
    assert!(matches!(body.block(entry).terminator(), Terminator::Jump { to, args } if *to == tail && args.is_empty()));
    assert!(matches!(body.block(tail).terminator(), Terminator::Diamond { join: j, .. } if *j == join));
    assert_eq!(body.block(tail).insts().len(), 1);
    // The moved instruction's definition and uses moved with it.
    rewriter.replace_all_uses(x, a);
    assert_eq!(rewriter.remove(x), Ok(()));
    assert!(matches!(rewriter.body().block(tail).insts()[0].0, Op::UnaryOp { dst, operand, .. } if dst == y && operand == a));
    // The Diamond's use of `c` moved with it to the tail's terminator.
    rewriter.replace_all_uses(c, d);
    assert!(matches!(rewriter.body().block(tail).terminator(), Terminator::Diamond { cond, .. } if *cond == d));
    assert_eq!(rewriter.body().use_count(c), 0);
}

// -- Builder: seal reports each kind of open state -----------------------

#[test]
fn seal_reports_a_block_without_its_terminator() {
    let mut f = fixture();
    let mut builder = f.module.build_body();
    let open = builder.new_block(&[]);
    builder.terminate(Terminator::Diverge);
    assert_eq!(builder.seal().err(), Some(Unsealed::BlockWithoutTerminator(open)));
}

#[test]
fn seal_reports_an_edge_to_an_unmade_block() {
    let mut f = fixture();
    let mut builder = f.module.build_body();
    let unmade = builder.reserve_block();
    builder.terminate(Terminator::Jump { to: unmade, args: Vec::new() });
    assert_eq!(builder.seal().err(), Some(Unsealed::EdgeToUnmadeBlock(unmade)));
}

#[test]
fn seal_reports_an_open_anyorder_scope() {
    let mut f = fixture();
    let mut builder = f.module.build_body();
    builder.open_anyorder();
    builder.terminate(Terminator::Diverge);
    assert_eq!(builder.seal().err(), Some(Unsealed::OpenAnyorder));
}

#[test]
fn a_sealed_body_is_added_to_the_module() {
    let mut f = fixture();
    let mut builder = f.module.build_body();
    builder.open_anyorder();
    builder.close_anyorder();
    builder.terminate(Terminator::Diverge);
    let id = builder.seal().expect("the body has no open state");
    f.module.set_entry(id);
    assert_eq!(f.module.entry(), id);
    assert!(matches!(f.body(id).block(f.body(id).entry()).terminator(), Terminator::Diverge));
}

#[test]
#[should_panic(expected = "exactly when it takes one")]
fn a_body_without_an_order_parameter_returns_no_token() {
    let mut f = fixture();
    let (na, nb, int, order) = (f.name("a"), f.name("o"), f.int, f.order);
    let mut builder = f.module.build_body();
    let a = builder.param(na, int);
    let o = builder.param(nb, order);
    builder.terminate(Terminator::Return { value: a, order: Some(o), span: Span::ZERO });
}
