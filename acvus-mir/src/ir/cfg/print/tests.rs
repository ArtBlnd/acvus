//! The printer over one body built through the Builder, each expected text
//! written out from the module doc's rules.

use acvus_ast::{Literal, Span};
use acvus_utils::{Astr, Interner, LocalIdOps, QualifiedRef};

use super::{Annotate, Names, View, body, module};
use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::cfg::{BlockId, BodyId, Module, Op, Terminator};
use crate::ir::{BinOp, Callee, PathSeg, RefTarget, ValOrigin};
use crate::passes::analysis::{
    Analysis, AnalysisManager, InitAnnotations, Initialization, LiveAnnotations, Liveness, LoanAnnotations, Loans,
};
use crate::ty::{
    Alignment, Calls, Concrete, Contexts, Cost, Effect, EffectTerm, Flow, FlowEnd, FlowTerm, Flows, Holds, IdentityId,
    IdentityTerm, IntTy, Mutability, ObjectTy, Reissue, SplitCost, Ty, TyKind, TypeArg, Types,
};

struct Fixture {
    module: Module,
    names: Interner,
    int: Ty,
    bool: Ty,
    /// `{a: i64, b: i64}`.
    object: Ty,
    mut_int: Ty,
    /// `fn() -> i64`, opaque, so a call of it is ordered.
    effectful: Ty,
}

fn fixture() -> Fixture {
    let names = Interner::new();
    let mut types = Types::<Concrete>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let bool = types.intern(TyKind::Bool);
    let (a, b) = (names.intern("a"), names.intern("b"));
    let object = types.intern(TyKind::Object(ObjectTy::Structural(Box::new([(a, int), (b, int)]), ())));
    let mut_int = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(int)));
    let effectful = types.intern(TyKind::Fn {
        params: Box::new([]),
        ret: int,
        effect: EffectTerm::Known(Effect::opaque()),
        flows: FlowTerm::Known(Flows::none()),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    Fixture { module: Module::new(types), names, int, bool, object, mut_int, effectful }
}

fn at(start: usize, end: usize) -> Span {
    Span::new(start, end)
}

impl Fixture {
    fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    fn field(&self, name: &str) -> Vec<PathSeg> {
        vec![PathSeg::Field(self.name(name))]
    }

    /// ```text
    /// x = {a: 1, b: 1}; y = x.a; r = &mut x.b; f(); if y < a { y } else { a }
    /// ```
    /// With `extra`, a constant is made first in the entry and removed
    /// after the seal: the same body, its values of other ids.
    fn build(&mut self, extra: bool) -> BodyId {
        let (int, bool, object, mut_int, effectful) = (self.int, self.bool, self.object, self.mut_int, self.effectful);
        let (a_field, b_field) = (self.field("a"), self.field("b"));
        let (param_name, x, y, f) = (self.name("a"), self.name("x"), self.name("y"), self.name("f"));
        let mut builder = self.module.build_body(true, int);
        let a = builder.param(param_name, int, at(0, 1));
        let slot = builder.slot(object, at(3, 18));
        let spare = extra.then(|| builder.push(Op::Const { value: Literal::Int(99) }, int, at(19, 20)));
        let one = builder.push(Op::Const { value: Literal::Int(1) }, int, at(20, 21));
        let assign = |path: Vec<PathSeg>| Op::Assign { target: RefTarget::Var(slot), path, value: one, restores: false };
        builder.push_unit(assign(a_field.clone()), at(22, 27));
        builder.push_unit(assign(b_field.clone()), at(28, 33));
        let read = builder.push(Op::Take { target: RefTarget::Var(slot), path: a_field, taken_out: false }, int, at(34, 40));
        builder.push(Op::Ref { target: RefTarget::Var(slot), path: b_field, mutability: Mutability::Mut }, mut_int, at(41, 47));
        let call = Op::FunctionCall { callee: Callee::Direct(QualifiedRef::root(f)), callee_ty: effectful, args: Vec::new() };
        let called = builder.push_effect(call, int, at(48, 51));
        let cond = builder.push(Op::BinOp { op: BinOp::Lt, left: read, right: a }, bool, at(52, 57));
        let then_block = builder.new_block(&[int]);
        let else_block = builder.new_block(&[]);
        let join = builder.new_block(&[int]);
        let diamond = Terminator::Diamond {
            cond,
            then_to: then_block,
            then_args: vec![read],
            else_to: else_block,
            else_args: Vec::new(),
            join,
        };
        builder.terminate(diamond, at(58, 60));
        builder.switch_to(then_block);
        let then_value = builder.block_params(then_block)[0];
        builder.terminate(Terminator::Jump { to: join, args: vec![then_value] }, at(61, 62));
        builder.switch_to(else_block);
        builder.terminate(Terminator::Jump { to: join, args: vec![a] }, at(63, 64));
        builder.switch_to(join);
        let joined = builder.block_params(join)[0];
        builder.terminate(Terminator::Return { value: joined, order: None, span: at(65, 66) }, at(65, 66));
        let id = builder.seal().expect("the body has no open state");

        if let Some(spare) = spare {
            let mut rewriter = self.module.rewrite(id);
            rewriter.remove(spare).expect("the spare constant has no use");
            rewriter.finish();
        }
        let origins = &mut self.module.bodies[id.0 as usize].debug.val_origins;
        origins.insert(slot, ValOrigin::Named(x));
        origins.insert(read, ValOrigin::Named(y));
        origins.insert(called, ValOrigin::Call(f));
        id
    }

    fn facts<A>(&self, body: BodyId) -> std::rc::Rc<A::Facts>
    where
        A: Analysis,
    {
        let decls = Declarations::verify(RawDeclarations {
            lang_impls: Vec::new(),
            types: Types::new(),
            functions: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            entries: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(&self.names),
        })
        .expect("empty declarations verify");
        let mut am = AnalysisManager::new(&decls);
        am.get::<A>(&self.module, body)
    }
}

const CFG: &str = "\
body0 entry=b0
b0():
  v3: i64 = Const 1
  Assign slot0.a, v3
  Assign slot0.b, v3
  v4: i64 = Take slot0.a
  v5: &mut i64 = Ref(mut) slot0.b
  v6: i64 = Call f()
  v8: bool = Lt v4, p0
  Diamond v8 ? b1(v4) : b2()
b1(v9: i64):
  Jump b3(v9)
b2():
  Jump b3(p0)
b3(v10: i64):
  Return v10
";

const FULL: &str = "\
body0 (p0 a: i64) order=t0 -> i64 entry=b0
slot0 x: {a: i64, b: i64} @3..18
b0():
  v3: i64 = Const 1 @20..21
  Assign slot0.a, v3 @22..27
  Assign slot0.b, v3 @28..33
  v4 y: i64 = Take slot0.a @34..40
  v5: &mut i64 = Ref(mut) slot0.b @41..47
  t7, v6 f(): i64 = Call f() t0 @48..51
  v8: bool = Lt v4, p0 @52..57
  Diamond v8 ? b1(v4) : b2() join b3 @58..60
b1(v9: i64):
  Jump b3(v9) @61..62
b2():
  Jump b3(p0) @63..64
b3(v10: i64):
  Return v10 t7 @65..66
";

/// Blocks in reverse postorder, `b0, b2, b1, b3` by their own ids, and
/// values numbered where they print.
const CANONICAL: &str = "\
body0 entry=b0
b0():
  v0: i64 = Const 1
  Assign slot0.a, v0
  Assign slot0.b, v0
  v1: i64 = Take slot0.a
  v2: &mut i64 = Ref(mut) slot0.b
  v3: i64 = Call f()
  v4: bool = Lt v1, p0
  Diamond v4 ? b2(v1) : b1()
b1():
  Jump b3(p0)
b2(v5: i64):
  Jump b3(v5)
b3(v6: i64):
  Return v6
";

#[test]
fn a_body_prints_its_blocks_values_and_types_in_the_cfg_view() {
    let mut f = fixture();
    let id = f.build(false);
    assert_eq!(body(&f.module, id, &f.names, View::CFG, &[]), CFG);
}

#[test]
fn the_full_view_adds_the_header_spans_order_structure_and_origins() {
    let mut f = fixture();
    let id = f.build(false);
    assert_eq!(body(&f.module, id, &f.names, View::FULL, &[]), FULL);
}

#[test]
fn a_canonical_print_is_the_same_for_two_modules_that_differ_only_in_ids() {
    let (mut plain, mut renumbered) = (fixture(), fixture());
    plain.build(false);
    renumbered.build(true);
    let canonical = View { canonical: true, ..View::CFG };
    assert_ne!(
        module(&plain.module, &plain.names, View::CFG),
        module(&renumbered.module, &renumbered.names, View::CFG),
        "by their own ids, the removed constant shifts every later value"
    );
    assert_eq!(module(&plain.module, &plain.names, canonical), CANONICAL);
    assert_eq!(module(&renumbered.module, &renumbered.names, canonical), CANONICAL);
}

#[test]
fn printing_twice_gives_the_same_text() {
    let mut f = fixture();
    f.build(false);
    f.build(true);
    for view in [View::CFG, View::FULL, View { canonical: true, ..View::FULL }] {
        assert_eq!(module(&f.module, &f.names, view), module(&f.module, &f.names, view));
    }
    assert_eq!(
        module(&f.module, &f.names, View::CFG),
        format!("{CFG}\n{}", body(&f.module, BodyId(1), &f.names, View::CFG, &[])),
        "a module is its bodies by id, one after another"
    );
}

#[test]
fn live_annotations_state_the_values_live_at_each_block_start() {
    let mut f = fixture();
    let id = f.build(false);
    let liveness = f.facts::<Liveness>(id);
    let live = LiveAnnotations::new(&liveness);
    let expected = "\
body0 entry=b0
# live: {p0, slot0, t0}
b0():
  v3: i64 = Const 1
  Assign slot0.a, v3
  Assign slot0.b, v3
  v4: i64 = Take slot0.a
  v5: &mut i64 = Ref(mut) slot0.b
  v6: i64 = Call f()
  v8: bool = Lt v4, p0
  Diamond v8 ? b1(v4) : b2()
# live: {t7, v9}
b1(v9: i64):
  Jump b3(v9)
# live: {p0, t7}
b2():
  Jump b3(p0)
# live: {t7, v10}
b3(v10: i64):
  Return v10
";
    assert_eq!(body(&f.module, id, &f.names, View::CFG, &[&live]), expected);
}

#[test]
fn init_annotations_state_each_unit_before_each_instruction() {
    let mut f = fixture();
    let id = f.build(false);
    let initialization = f.facts::<Initialization>(id);
    let init = InitAnnotations::new(&f.names, &initialization);
    let expected = "\
body0 entry=b0
b0():
  # init: slot0.a=N slot0.b=N
  v3: i64 = Const 1
  # init: slot0.a=N slot0.b=N
  Assign slot0.a, v3
  # init: slot0.a=W slot0.b=N
  Assign slot0.b, v3
  # init: slot0.a=W slot0.b=W
  v4: i64 = Take slot0.a
  # init: slot0.a=W slot0.b=W
  v5: &mut i64 = Ref(mut) slot0.b
  # init: slot0.a=W slot0.b=W
  v6: i64 = Call f()
  # init: slot0.a=W slot0.b=W
  v8: bool = Lt v4, p0
  Diamond v8 ? b1(v4) : b2()
b1(v9: i64):
  Jump b3(v9)
b2():
  Jump b3(p0)
b3(v10: i64):
  Return v10
";
    assert_eq!(body(&f.module, id, &f.names, View::CFG, &[&init]), expected);
}

#[test]
fn loan_annotations_state_the_loans_each_defined_value_holds() {
    let mut f = fixture();
    let id = f.build(false);
    let loans = f.facts::<Loans>(id);
    let loan_annotations = LoanAnnotations::new(f.module.body(id), &f.names, &loans);
    let expected = "\
body0 entry=b0
b0():
  v3: i64 = Const 1
  Assign slot0.a, v3
  Assign slot0.b, v3
  v4: i64 = Take slot0.a
  # loans: v5=[slot0.b mut @41..47]
  v5: &mut i64 = Ref(mut) slot0.b
  v6: i64 = Call f()
  v8: bool = Lt v4, p0
  Diamond v8 ? b1(v4) : b2()
b1(v9: i64):
  Jump b3(v9)
b2():
  Jump b3(p0)
b3(v10: i64):
  Return v10
";
    assert_eq!(body(&f.module, id, &f.names, View::CFG, &[&loan_annotations]), expected);
}

/// Annotations at one point print in key order, whichever annotator gives
/// them.
#[test]
fn annotations_at_one_point_print_sorted_by_key() {
    struct Says(&'static str);
    impl Annotate for Says {
        fn block(&self, _: &Names<'_>, _: BlockId) -> Vec<(&'static str, String)> {
            vec![(self.0, "1".to_string())]
        }
        fn before(&self, _: &Names<'_>, _: crate::ir::cfg::InstId) -> Vec<(&'static str, String)> {
            Vec::new()
        }
        fn edge(&self, _: &Names<'_>, _: BlockId, _: BlockId) -> Vec<(&'static str, String)> {
            Vec::new()
        }
    }
    let mut f = fixture();
    let id = f.build(false);
    let (later, earlier) = (Says("b"), Says("a"));
    let text = body(&f.module, id, &f.names, View::CFG, &[&later, &earlier]);
    assert!(text.starts_with("body0 entry=b0\n# a: 1\n# b: 1\nb0():\n"), "{text}");
}

/// With `canonical` on, an annotation names values and blocks as the print
/// does: the body built with a value made and removed prints the same text
/// as the plain one, annotations included.
#[test]
fn annotations_name_values_and_blocks_as_a_canonical_print_does() {
    let expected = "\
body0 entry=b0
# live: {p0, slot0, t0}
b0():
  v0: i64 = Const 1
  Assign slot0.a, v0
  Assign slot0.b, v0
  v1: i64 = Take slot0.a
  # loans: v2=[slot0.b mut @41..47]
  v2: &mut i64 = Ref(mut) slot0.b
  v3: i64 = Call f()
  v4: bool = Lt v1, p0
  Diamond v4 ? b2(v1) : b1()
# live: {p0, t1}
b1():
  Jump b3(p0)
# live: {t1, v5}
b2(v5: i64):
  Jump b3(v5)
# live: {t1, v6}
b3(v6: i64):
  Return v6
";
    let canonical = View { canonical: true, ..View::CFG };
    for extra in [false, true] {
        let mut f = fixture();
        let id = f.build(extra);
        let liveness = f.facts::<Liveness>(id);
        let loans = f.facts::<Loans>(id);
        let live = LiveAnnotations::new(&liveness);
        let loan_annotations = LoanAnnotations::new(f.module.body(id), &f.names, &loans);
        assert_eq!(body(&f.module, id, &f.names, canonical, &[&live, &loan_annotations]), expected, "extra: {extra}");
    }
}

/// A function type's effect, flows, calls and holds, and a declared type's
/// identity, print in a bracket where they differ from their defaults
/// (types §5), and not at all where they equal them. A declared type's
/// effect prints after its type arguments as `with` and the effect, wherever
/// it carries one (extern-fn §8).
#[test]
fn a_type_prints_what_differs_from_its_defaults_in_a_bracket() {
    let names = Interner::new();
    let mut types = Types::<Concrete>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let ref_int = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    let (c, d) = (QualifiedRef::root(names.intern("c")), QualifiedRef::root(names.intern("d")));
    let mut function = |params: &[Ty], ret: Ty, effect: Effect, flows: Flows, calls: Calls, holds: Holds| {
        types.intern(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(flows),
            calls,
            holds,
            values: (),
        })
    };
    let defaults = function(&[int], int, Effect::opaque(), Flows::Every, Calls::Many, Holds::Nothing);
    let pure_once = function(&[int], int, Effect::pure(), Flows::Every, Calls::Once, Holds::Nothing);
    let reading = Effect::new(Reissue::Idempotent, true, Contexts::from([c]), Contexts::new(), SplitCost::WithCost(Cost::stated(3)));
    let flows = Flows::of([
        Flow { to: FlowEnd::Result, from: FlowEnd::Captures, alignment: Alignment::Any },
        Flow { to: FlowEnd::Result, from: FlowEnd::Param(0), alignment: Alignment::Aligned },
    ]);
    let everything = function(&[ref_int], ref_int, reading, flows, Calls::Many, Holds::Mut);
    let writing = Effect::new(Reissue::Opaque, true, Contexts::new(), Contexts::from([d]), SplitCost::Unavailable);
    let no_flows = function(&[], int, writing, Flows::none(), Calls::Many, Holds::Shared);
    let vec = |types: &mut Types<Concrete>, effect: Option<EffectTerm<Concrete>>, identity: Option<IdentityTerm<Concrete>>| {
        types.intern(TyKind::UserDefined {
            id: QualifiedRef::root(names.intern("Vec")),
            type_args: Box::new([TypeArg::Specialized(int)]),
            effect,
            identity,
        })
    };
    let plain_vec = vec(&mut types, None, None);
    let opaque_vec = vec(&mut types, Some(EffectTerm::Known(Effect::opaque())), None);
    let pure_vec = vec(&mut types, Some(EffectTerm::Known(Effect::pure())), None);
    let marked_vec = vec(
        &mut types,
        Some(EffectTerm::Known(pure_reading(c))),
        Some(IdentityTerm::Known(IdentityId::from_raw(3))),
    );

    let shown = |ty| crate::ty::print::ty(&types, &names, ty);
    assert_eq!(shown(defaults), "fn(i64) -> i64");
    assert_eq!(shown(pure_once), "fn(i64) -> i64 [pure, calls=once]");
    assert_eq!(
        shown(everything),
        "fn(&i64) -> &i64 [idempotent, commutative, reads=[@c], split=3, flows=[p0->ret, cap->ret:any], holds=mut]"
    );
    assert_eq!(shown(no_flows), "fn() -> i64 [writes=[@d], flows=[], holds=shared]");
    assert_eq!(shown(plain_vec), "Vec<#i64>");
    assert_eq!(shown(opaque_vec), "Vec<#i64> with opaque");
    assert_eq!(shown(pure_vec), "Vec<#i64> with pure");
    assert_eq!(shown(marked_vec), "Vec<#i64> with (pure, reads=[@c]) [id0=3]");
}

fn pure_reading(context: QualifiedRef) -> Effect {
    Effect::new(Reissue::Pure, true, Contexts::from([context]), Contexts::new(), SplitCost::Unavailable)
}

/// An extern call prints its impl, `#n`, where the function has impls and
/// the call is at impl `n`; a function without impls prints its name
/// alone. A name prints as its namespace's path joined by `::`.
#[test]
fn an_extern_call_prints_its_impl_where_its_function_has_impls() {
    let mut f = fixture();
    let (int, bool) = (f.int, f.bool);
    let array = f.module.types.intern(TyKind::Array(int, crate::ty::LenTerm::Known(2), ()));
    let array_ref = f.module.types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(array)));
    let slice = f.module.types.intern(TyKind::Slice(int));
    let slice_ref = f.module.types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(slice)));
    let pure = |types: &mut Types<Concrete>, params: &[Ty], ret: Ty| {
        types.intern(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        })
    };
    let eq_ty = pure(&mut f.module.types, &[int, int], bool);
    let g_ty = pure(&mut f.module.types, &[int], int);
    let core = f.names.namespace(&[f.name("core")]);
    let (eq, g, as_slice) =
        (QualifiedRef::qualified(core, f.name("eq")), QualifiedRef::root(f.name("g")), QualifiedRef::qualified(core, f.name("as_slice")));
    let (n, xs) = (f.name("n"), f.name("xs"));

    let mut builder = f.module.build_body(false, int);
    let n = builder.param(n, int, Span::ZERO);
    let xs = builder.param(xs, array_ref, Span::ZERO);
    let at_impl = Op::FunctionCall { callee: Callee::Extern { id: eq, r#impl: Some(1), required: Vec::new() }, callee_ty: eq_ty, args: vec![n, n] };
    builder.push(at_impl, bool, Span::ZERO);
    let at_none = Op::FunctionCall { callee: Callee::Extern { id: g, r#impl: None, required: Vec::new() }, callee_ty: g_ty, args: vec![n] };
    let result = builder.push(at_none, int, Span::ZERO);
    let view = Op::AsSlice { container: xs, mutability: Mutability::Shared, r#impl: crate::ir::ExternImpl { id: as_slice, r#impl: Some(0) } };
    builder.push(view, slice_ref, Span::ZERO);
    builder.terminate(Terminator::Return { value: result, order: None, span: Span::ZERO }, Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let expected = "\
body0 entry=b0
b0():
  v2: bool = Call core::eq#1(p0, p0)
  v3: i64 = Call g(p0)
  v4: &[i64] = AsSlice core::as_slice#0 p1
  Return v3
";
    assert_eq!(body(&f.module, id, &f.names, View::CFG, &[]), expected);
}
