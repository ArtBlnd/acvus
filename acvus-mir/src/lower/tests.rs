//! Each form, from source to IR: parsed with `acvus-ast`, checked by
//! typeck over declarations that went through `verify`, lowered, and every
//! body held to the editor's invariants.

use acvus_ast::Script;
use acvus_utils::{Astr, Interner, QualifiedRef};

use super::{lower_script, lower_template};
use crate::decl::{
    ContextDecl, ConversionRule, Core, Declarations, DeclaredFacts, ImplSig, Impls, LoopMarkers, RawDeclarations, Scheme, TyVarBound, UserTypeDecl,
};
use crate::ir::cfg::edit::{DefDominatesUses, Invariant, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree, Violation};
use crate::ir::cfg::{Body, BodyId, Module, Op, OrderUse, Terminator};
use crate::ir::{Callee, ForSource, PathSeg, RefTarget, SwitchKey};
use crate::passes::analysis::{Loan, Mode, Storage};
use crate::ty::{Calls, Concrete, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Poly, TyId, TyKind, TypeArg, Types};
use crate::typeck::{check_script, check_template};

struct Fixture {
    names: Interner,
    raw: RawDeclarations,
}

impl Fixture {
    fn new() -> Self {
        let names = Interner::new();
        let core = Core::of(&names);
        Fixture {
            names,
            raw: RawDeclarations {
                types: Types::new(),
                functions: Vec::new(),
                lang_impls: Vec::new(),
                user_types: Vec::new(),
                casts: Vec::new(),
                conversions: Vec::new(),
                contexts: Vec::new(),
                inputs: Vec::new(),
                loop_markers: LoopMarkers::default(),
                core,
            },
        }
    }

    fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    fn int(&mut self) -> TyId<Poly> {
        self.ty(TyKind::Int(IntTy::I64))
    }

    fn reference(&mut self, mutability: Mutability, target: TyId<Poly>) -> TyId<Poly> {
        self.ty(TyKind::Ref(mutability, TypeArg::Uniform(target)))
    }

    fn function(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>, effect: Effect) -> TyId<Poly> {
        self.ty(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        })
    }

    fn param_names(&self, count: usize) -> Box<[Astr]> {
        (0..count).map(|at| self.name(&format!("p{at}"))).collect()
    }

    /// A plain extern fn `name` of type `ty`, with no type variable.
    fn declare(&mut self, name: &str, ty: TyId<Poly>, arity: usize) -> QualifiedRef {
        let id = QualifiedRef::root(self.name(name));
        let scheme = Scheme { ty, param_names: self.param_names(arity), bounds: Box::new([]), impls: None, requires: Box::new([]), facts: DeclaredFacts::default() };
        self.raw.functions.push((id, scheme));
        id
    }

    /// `f(i64) -> i64`, whose effect is opaque.
    fn opaque_f(&mut self) -> QualifiedRef {
        let int = self.int();
        let ty = self.function(&[int], int, Effect::opaque());
        self.declare("f", ty, 1)
    }

    fn input(&mut self, name: &str, ty: TyId<Poly>) {
        let name = self.name(name);
        self.raw.inputs.push((name, ty));
    }

    fn context(&mut self, name: &str, ty: TyId<Poly>, volatile: bool) -> QualifiedRef {
        let id = QualifiedRef::root(self.name(name));
        self.raw.contexts.push((id, ContextDecl { ty, volatile }));
        id
    }

    fn lang_item(&mut self, item: QualifiedRef, impls: &[TyId<Poly>]) {
        let concrete = impls
            .iter()
            .map(|ty| ImplSig {
                ty: *ty,
                param_names: self.param_names(match self.raw.types.kind(*ty) {
                    TyKind::Fn { params, .. } => params.len(),
                    _ => 0,
                }),
                requires: Box::new([]),
                facts: Default::default(),
            })
            .collect();
        self.raw.lang_impls.push((item, Impls { concrete, generic: None }));
    }

    /// An extern `Vec<T>` whose `as_slice` impl views it as `&[T]`, and the
    /// type `Vec<i64>`.
    fn vec_of_ints(&mut self) -> TyId<Poly> {
        let vec_id = QualifiedRef::root(self.name("Vec"));
        self.raw.user_types.push(UserTypeDecl {
            id: vec_id,
            type_params: Box::new([TyVarBound::Any]),
            specializes: Box::new([false]),
            effect_params: 0,
            identity_params: 0,
            lifetimes: 0,
        });
        let e = self.ty(TyKind::Var(0));
        let vec_e = self.vec(vec_id, e);
        let shared_vec_e = self.reference(Mutability::Shared, vec_e);
        let slice_e = self.ty(TyKind::Slice(e));
        let shared_slice_e = self.reference(Mutability::Shared, slice_e);
        let view = self.function(&[shared_vec_e], shared_slice_e, Effect::pure());
        let as_slice = self.raw.core.as_slice;
        self.lang_item(as_slice, &[view]);
        let int = self.int();
        self.vec(vec_id, int)
    }

    fn vec(&mut self, vec_id: QualifiedRef, element: TyId<Poly>) -> TyId<Poly> {
        self.ty(TyKind::UserDefined {
            id: vec_id,
            type_args: Box::new([TypeArg::Uniform(element)]),
            effect_args: Box::new([]),
            identity_args: Box::new([]),
        })
    }

    /// `eq(&w, &w) -> bool` and `cmp(&w, &w) -> i64` at `i64` and `bool`.
    fn comparisons(&mut self) {
        let int = self.int();
        let bool_ty = self.ty(TyKind::Bool);
        let mut eqs = Vec::new();
        let mut cmps = Vec::new();
        for w in [int, bool_ty] {
            let shared = self.reference(Mutability::Shared, w);
            eqs.push(self.function(&[shared, shared], bool_ty, Effect::pure()));
            cmps.push(self.function(&[shared, shared], int, Effect::pure()));
        }
        let (eq, cmp) = (self.raw.core.eq, self.raw.core.cmp);
        self.lang_item(eq, &eqs);
        self.lang_item(cmp, &cmps);
    }

    /// `display(&i64, &mut String)`, pure.
    fn display_ints(&mut self) {
        let int = self.int();
        let shown = self.reference(Mutability::Shared, int);
        let string = self.ty(TyKind::String);
        let out = self.reference(Mutability::Mut, string);
        let unit = self.ty(TyKind::Unit);
        let impl_ty = self.function(&[shown, out], unit, Effect::pure());
        let display = self.raw.core.display;
        self.lang_item(display, &[impl_ty]);
    }

    fn script(self, source: &str) -> Lowered {
        let script: Script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let typeck = check_script(&decls, &mut types, &script);
        assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
        let module = lower_script(&decls, &typeck, types, &script);
        Lowered::checked(module, self.names)
    }

    fn template(self, source: &str) -> Lowered {
        let template = acvus_ast::parse_template(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let typeck = check_template(&decls, &mut types, &template);
        assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
        let module = lower_template(&decls, &typeck, types, &template);
        Lowered::checked(module, self.names)
    }
}

/// A lowered module whose every body holds the editor's invariants.
struct Lowered {
    module: Module,
    names: Interner,
}

type Checker = fn(&Body, &Types<Concrete>) -> Result<(), Violation>;

impl Lowered {
    fn checked(module: Module, names: Interner) -> Lowered {
        let checkers: [Checker; 5] = [
            <TypesAgree as Invariant>::check,
            <DefDominatesUses as Invariant>::check,
            <OrderChainWhole as Invariant>::check,
            <StructureKept as Invariant>::check,
            <SplitTakenOnce as Invariant>::check,
        ];
        for id in module.body_ids() {
            for check in checkers {
                if let Err(violation) = check(module.body(id), module.types()) {
                    panic!("body {id:?} breaks {violation:?}\n{}", shown(module.body(id)));
                }
            }
        }
        Lowered { module, names }
    }

    fn entry(&self) -> &Body {
        self.module.body(self.module.entry())
    }

    fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    fn ops(&self, body: BodyId) -> Vec<&Op> {
        self.module.body(body).blocks().flat_map(|(_, block)| block.insts()).map(|inst| inst.op()).collect()
    }

    fn entry_ops(&self) -> Vec<&Op> {
        self.ops(self.module.entry())
    }

    fn terminators(&self, body: BodyId) -> Vec<&Terminator> {
        self.module.body(body).blocks().map(|(_, block)| block.terminator()).collect()
    }

    fn entry_terminators(&self) -> Vec<&Terminator> {
        self.terminators(self.module.entry())
    }

    fn kind_of(&self, body: &Body, value: crate::ir::ValueId) -> &TyKind<Concrete> {
        self.module.types().kind(body.ty_of(value))
    }
}

/// A body, block by block, for a failing test's message.
fn shown(body: &Body) -> String {
    let mut out = String::new();
    for (id, block) in body.blocks() {
        out.push_str(&format!("{id:?}({:?}):\n", block.params()));
        for inst in block.insts() {
            out.push_str(&format!("  {:?} = {:?} {:?}\n", inst.result(), inst.op(), inst.order()));
        }
        out.push_str(&format!("  {:?}\n", block.terminator()));
    }
    out
}

fn count(ops: &[&Op], wanted: impl Fn(&Op) -> bool) -> usize {
    ops.iter().filter(|op| wanted(op)).count()
}

// -- control ---------------------------------------------------------------

/// `if` is a `Diamond` whose join takes the value (control-flow, D45).
#[test]
fn if_is_a_diamond_whose_join_takes_the_value() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let lowered = fx.script("if $c { 1 } else { 2 }");
    let body = lowered.entry();
    let joins: Vec<_> = lowered
        .entry_terminators()
        .into_iter()
        .filter_map(|terminator| match terminator {
            Terminator::Diamond { join, .. } => Some(*join),
            _ => None,
        })
        .collect();
    let [join] = joins[..] else { panic!("one Diamond: {}", shown(body)) };
    let [value] = body.block(join).params() else { panic!("the join takes the if's value") };
    assert!(matches!(lowered.kind_of(body, *value), TyKind::Int(IntTy::I64)));
}

/// `&&` evaluates its right side only where its left is true: a `Diamond`,
/// and no call (operators §2).
#[test]
fn and_is_a_short_circuit_diamond() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("a", bool_ty);
    fx.input("b", bool_ty);
    let lowered = fx.script("$a && $b");
    let terminators = lowered.entry_terminators();
    let diamond = terminators
        .iter()
        .find_map(|terminator| match terminator {
            Terminator::Diamond { else_to, .. } => Some(*else_to),
            _ => None,
        })
        .expect("a Diamond");
    let else_ops: Vec<_> = lowered.entry().block(diamond).insts().collect();
    assert!(matches!(else_ops[0].op(), Op::Const { value: acvus_ast::Literal::Bool(false) }));
    assert_eq!(count(&lowered.entry_ops(), |op| matches!(op, Op::FunctionCall { .. })), 0);
}

/// A `match` over one enum's tags is one `Switch` holding its join (D57).
#[test]
fn a_match_over_tags_is_one_switch() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int));
    fx.input("o", option);
    let lowered = fx.script("match $o { Some(x) => x, None => 0, }");
    let switches: Vec<_> = lowered
        .entry_terminators()
        .into_iter()
        .filter_map(|terminator| match terminator {
            Terminator::Switch { arms, default, join, .. } => Some((arms.clone(), default.clone(), *join)),
            _ => None,
        })
        .collect();
    let [(arms, default, join)] = &switches[..] else { panic!("one Switch: {}", shown(lowered.entry())) };
    let keys: Vec<SwitchKey> = arms.iter().map(|(key, _, _)| *key).collect();
    assert_eq!(keys, vec![SwitchKey::Tag(lowered.name("Some")), SwitchKey::Tag(lowered.name("None"))]);
    assert!(default.is_none());
    assert_eq!(lowered.entry().block(*join).params().len(), 1);
    assert_eq!(count(&lowered.entry_ops(), |op| matches!(op, Op::TestVariant { .. })), 0);
}

/// A `match` whose arm tests a nested pattern is tested arm by arm, in
/// order, each arm a `Diamond` to the one join (D57).
#[test]
fn a_nested_pattern_is_tested_arm_by_arm() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int));
    let pair = fx.ty(TyKind::Tuple(Box::new([option, int])));
    fx.input("p", pair);
    let lowered = fx.script("match $p { (Some(x), 1) => x, _ => 0, }");
    let terminators = lowered.entry_terminators();
    assert!(!terminators.iter().any(|terminator| matches!(terminator, Terminator::Switch { .. })));
    let joins: Vec<_> = terminators
        .iter()
        .filter_map(|terminator| match terminator {
            Terminator::Diamond { join, .. } => Some(*join),
            _ => None,
        })
        .collect();
    assert!(joins.len() >= 2, "the tests and the arm are Diamonds: {}", shown(lowered.entry()));
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::TestVariant { .. })), 1);
    assert_eq!(count(&ops, |op| matches!(op, Op::TestLiteral { .. })), 1);
}

/// A `for` over `&v` reaches the elements through the `as_slice` impl
/// typeck recorded, and traverses the slice (operators §3).
#[test]
fn a_for_over_a_reference_to_a_container_traverses_its_as_slice() {
    let mut fx = Fixture::new();
    let vec = fx.vec_of_ints();
    fx.input("v", vec);
    let lowered = fx.script("for x in &$v { x; } 1");
    let ops = lowered.entry_ops();
    let slice = lowered
        .entry()
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .find(|inst| matches!(inst.op(), Op::AsSlice { mutability: Mutability::Shared, .. }))
        .and_then(|inst| inst.result())
        .expect("an AsSlice");
    assert!(lowered.entry_terminators().iter().any(|terminator| matches!(
        terminator,
        Terminator::For { source: ForSource::Slice(source), .. } if *source == slice
    )));
    assert_eq!(count(&ops, |op| matches!(op, Op::FunctionCall { .. })), 0);
}

/// A `for` over a reference to an array traverses the language's own view
/// of it (extern-fn §1, D67 item 3).
#[test]
fn a_for_over_a_reference_to_an_array_traverses_its_view() {
    let lowered = Fixture::new().script("let a = [1, 2]; for x in &a { x; } 1");
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::ArrayView { mutability: Mutability::Shared, .. })), 1);
    assert!(lowered
        .entry_terminators()
        .iter()
        .any(|terminator| matches!(terminator, Terminator::For { source: ForSource::Slice(_), .. })));
}

/// A `for` over a range is a `For` of its bounds, whose body takes the
/// element alone.
#[test]
fn a_for_over_a_range_is_a_for_of_its_bounds() {
    let lowered = Fixture::new().script("for i in 0..3 { i; } 1");
    let body = lowered.entry();
    let stage = lowered
        .entry_terminators()
        .into_iter()
        .find_map(|terminator| match terminator {
            Terminator::For { source: ForSource::Range { .. }, stages, .. } => Some(stages.body()),
            _ => None,
        })
        .expect("a For over a range");
    assert_eq!(body.block(stage).params().len(), 1);
}

/// `while let` is a `While` whose condition is the pattern's test, the
/// names bound in its body.
#[test]
fn while_let_is_a_while_on_the_patterns_test() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int));
    fx.input("o", option);
    let lowered = fx.script("let o = $o; while let Some(x) = o { o = None; } 1");
    let body = lowered.entry();
    let (cond, stage) = lowered
        .entry_terminators()
        .into_iter()
        .find_map(|terminator| match terminator {
            Terminator::While { cond, stages, .. } => Some((*cond, stages.body())),
            _ => None,
        })
        .expect("a While");
    assert!(matches!(lowered.kind_of(body, cond), TyKind::Bool));
    let bound: Vec<_> = body.block(stage).insts().collect();
    assert!(
        matches!(bound[0].op(), Op::Take { path, .. } if *path == [PathSeg::Payload]),
        "the body binds x first, taking the payload out of the scrutinee's slot (D89): {}",
        shown(body)
    );
}

/// D89: `if let (Some(a), b) = pair()` puts the pair in a slot; the test
/// reads its first part through a reference and takes nothing, the bind
/// takes each part once, and where the test fails both parts are dropped
/// as parts. The pair is never taken or dropped whole.
#[test]
fn a_pattern_test_moves_nothing_and_a_bind_takes_each_part_once() {
    let mut fx = Fixture::new();
    let string = fx.ty(TyKind::String);
    let option = fx.ty(TyKind::Option(string));
    let pair = fx.ty(TyKind::Tuple(Box::new([option, string])));
    let pair_fn = fx.function(&[], pair, Effect::pure());
    fx.declare("pair", pair_fn, 0);
    let script: Script = acvus_ast::parse_script(&fx.names, "if let (Some(a), b) = pair() { 1 } else { 2 }").expect("parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let mut types = Types::new();
    let typeck = check_script(&decls, &mut types, &script);
    assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
    let mut module = lower_script(&decls, &typeck, types, &script);
    let mut am = crate::passes::analysis::AnalysisManager::new(&decls);
    super::required::insert_drops(&mut module, &mut am);
    let lowered = Lowered::checked(module, fx.names);
    let body = lowered.entry();
    let ops = lowered.entry_ops();
    let slot = ops
        .iter()
        .find_map(|op| match op {
            Op::Assign { target: RefTarget::Var(slot), path, .. }
                if path.is_empty() && matches!(lowered.kind_of(body, *slot), TyKind::Tuple(items) if items.len() == 2) =>
            {
                Some(*slot)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("the pair is put in a slot: {}", shown(body)));
    let takes = |wanted: &[PathSeg]| {
        count(&ops, |op| matches!(op, Op::Take { target: RefTarget::Var(named), path, .. } if *named == slot && path == wanted))
    };
    assert_eq!(takes(&[]), 0, "the pair is never taken whole: {}", shown(body));
    assert_eq!(takes(&[PathSeg::Index(0), PathSeg::Payload]), 1, "`a` is bound once: {}", shown(body));
    assert_eq!(takes(&[PathSeg::Index(1)]), 2, "`b` is bound once, and dropped once where the test fails: {}", shown(body));
    assert_eq!(takes(&[PathSeg::Index(0)]), 1, "the option is dropped once where the test fails: {}", shown(body));
    let tested = count(&ops, |op| matches!(op, Op::Ref { target: RefTarget::Var(named), path, .. } if *named == slot && *path == [PathSeg::Index(0)]));
    assert_eq!(tested, 1, "the test reads the option through a reference: {}", shown(body));
    assert!(ops.iter().any(|op| matches!(op, Op::TestVariant { .. })));
    assert!(!ops.iter().any(|op| matches!(op, Op::TupleIndex { .. } | Op::UnwrapVariant { .. })), "tests extract nothing: {}", shown(body));
}

/// `break` goes to the innermost loop's exit and `continue` to its header.
#[test]
fn break_and_continue_go_to_the_loops_exit_and_header() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    fx.input("d", bool_ty);
    let lowered = fx.script("while $c { if $d { break; }; continue; } 1");
    let body = lowered.entry();
    let (header, exit) = body
        .blocks()
        .find_map(|(id, block)| match block.terminator() {
            Terminator::While { exit, .. } => Some((id, *exit)),
            _ => None,
        })
        .expect("a While");
    let jumps: Vec<_> = lowered
        .entry_terminators()
        .into_iter()
        .filter_map(|terminator| match terminator {
            Terminator::Jump { to, .. } => Some(*to),
            _ => None,
        })
        .collect();
    assert!(jumps.contains(&exit), "a break to the exit: {}", shown(body));
    assert!(
        jumps.iter().filter(|to| **to == header).count() >= 2,
        "the entry and the continue jump to the header: {}",
        shown(body)
    );
}

/// `?` tests the tag and returns the `None` early.
#[test]
fn try_returns_none_early() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int));
    fx.input("o", option);
    let lowered = fx.script("let x = $o?; Some(x)");
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::TestVariant { .. })), 1);
    let returns = lowered
        .entry_terminators()
        .into_iter()
        .filter(|terminator| matches!(terminator, Terminator::Return { .. }))
        .count();
    assert_eq!(returns, 2);
    assert!(lowered.entry_terminators().iter().any(|terminator| matches!(terminator, Terminator::JumpIf { .. })));
}

// -- effects ---------------------------------------------------------------

fn call_edges(lowered: &Lowered) -> Vec<crate::ir::OrderEdge> {
    lowered
        .entry()
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .filter(|inst| matches!(inst.op(), Op::FunctionCall { .. }))
        .map(|inst| match inst.order() {
            Some(OrderUse::Edge(edge)) => edge,
            other => panic!("an opaque call stands on the chain, not {other:?}"),
        })
        .collect()
}

/// Effectful calls are chained in sequential order (effects §3).
#[test]
fn effectful_calls_are_chained_in_order() {
    let mut fx = Fixture::new();
    let f = fx.opaque_f();
    let lowered = fx.script("f(1); f(2); 3");
    let [first, second] = call_edges(&lowered)[..] else { panic!("two calls") };
    assert_eq!(Some(first.before), lowered.entry().order_param());
    assert_eq!(second.before, first.after);
    assert!(lowered.entry_ops().iter().any(|op| matches!(op, Op::FunctionCall { callee: Callee::Extern { id, .. }, .. } if *id == f)));
}

/// Inside `anyorder`, every call consumes the entry token, and what they
/// yield is merged (effects §3).
#[test]
fn anyorder_calls_consume_one_token_and_merge() {
    let mut fx = Fixture::new();
    fx.opaque_f();
    let lowered = fx.script("anyorder { f(1); f(2); } 3");
    let [first, second] = call_edges(&lowered)[..] else { panic!("two calls") };
    assert_eq!(first.before, second.before);
    assert!(count(&lowered.entry_ops(), |op| matches!(op, Op::Merge { .. })) >= 1);
}

// -- closures ----------------------------------------------------------------

/// A lambda without `move` captures by reference, passing `&T`; one with
/// `move` passes the value (functions §2). Each is a body of its own.
#[test]
fn a_lambda_captures_by_reference_and_by_move() {
    let lowered = Fixture::new().script("let a = 1; let g = |x| -> a; let b = 2; let h = move |x| -> b; g(1) + 0; h(1)");
    lowered_closures(&lowered);
}

fn lowered_closures(lowered: &Lowered) {
    let closures: Vec<(BodyId, Vec<crate::ir::ValueId>)> = lowered
        .entry_ops()
        .into_iter()
        .filter_map(|op| match op {
            Op::MakeClosure { body, captures } => Some((*body, captures.clone())),
            _ => None,
        })
        .collect();
    let [(by_ref, by_ref_captures), (by_move, by_move_captures)] = &closures[..] else {
        panic!("two closures: {}", shown(lowered.entry()))
    };
    let entry = lowered.entry();
    assert!(matches!(lowered.kind_of(entry, by_ref_captures[0]), TyKind::Ref(Mutability::Shared, _)));
    assert!(matches!(lowered.kind_of(entry, by_move_captures[0]), TyKind::Int(IntTy::I64)));
    assert_eq!(lowered.module.body(*by_ref).captures().len(), 1);
    assert_eq!(lowered.module.body(*by_move).captures().len(), 1);
}

// -- contexts ----------------------------------------------------------------

/// A context not volatile is loaded at entry into its slot, committed and
/// loaded again around an opaque call, and committed at return
/// (context §3).
#[test]
fn a_context_is_loaded_at_entry_bracketed_around_a_call_and_committed_at_return() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let n = fx.context("n", int, false);
    fx.opaque_f();
    let lowered = fx.script("let a = @n; f(1); @n = a; 1");
    let ops = lowered.entry_ops();
    let shape: Vec<&str> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Fetch { context } if *context == n => Some("load"),
            Op::Commit { context, .. } if *context == n => Some("commit"),
            Op::FunctionCall { .. } => Some("call"),
            _ => None,
        })
        .collect();
    assert_eq!(shape, vec!["load", "commit", "call", "load", "commit"]);
    let slots = lowered.entry().slots();
    assert_eq!(slots.iter().filter(|slot| lowered.entry().context_of_slot(**slot) == Some(n)).count(), 1);
}

/// A volatile context has no slot: a read is a load and a write a commit
/// (context §3).
#[test]
fn a_volatile_context_is_a_load_at_each_read_and_a_commit_at_each_write() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let clock = fx.context("clock", int, true);
    let lowered = fx.script("let a = @clock; let b = @clock; @clock = a; 1");
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::Fetch { context } if *context == clock)), 2);
    assert_eq!(count(&ops, |op| matches!(op, Op::Commit { context, .. } if *context == clock)), 1);
    let body = lowered.entry();
    assert!(body.slots().iter().all(|slot| body.context_of_slot(*slot).is_none()));
}

// -- templates and conversions ------------------------------------------------

/// A template appends its text as a literal and `{{ x }}` through the
/// `core::display` impl onto one `String` it returns (programs §1).
#[test]
fn a_template_appends_text_and_displays_values_onto_its_output() {
    let mut fx = Fixture::new();
    let int = fx.int();
    fx.input("x", int);
    fx.display_ints();
    let lowered = fx.template("hello {{ $x }}");
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::ConstStr { text } if text == "hello ")), 1);
    assert_eq!(count(&ops, |op| matches!(op, Op::StringAppend { .. })), 1);
    let display = QualifiedRef::qualified(lowered.name("core"), lowered.name("display"));
    assert_eq!(
        count(&ops, |op| matches!(op, Op::FunctionCall { callee: Callee::Extern { id, .. }, .. } if *id == display)),
        1
    );
    assert!(matches!(lowered.module.types().kind(lowered.entry().ret()), TyKind::String));
}

/// A coercion is its chain of steps in order: `&Vec<i64>` passed where
/// `&[i64]` is taken goes through the view's `as_slice` impl before the
/// call (operators §5, types §7).
#[test]
fn a_view_step_of_a_coercion_is_its_as_slice() {
    let mut fx = Fixture::new();
    let vec = fx.vec_of_ints();
    fx.input("v", vec);
    let int = fx.int();
    let slice = fx.ty(TyKind::Slice(int));
    let shared_slice = fx.reference(Mutability::Shared, slice);
    let first_ty = fx.function(&[shared_slice], int, Effect::pure());
    fx.declare("first", first_ty, 1);
    let lowered = fx.script("first(&$v)");
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let view = insts.iter().position(|inst| matches!(inst.op(), Op::AsSlice { .. })).expect("the view's AsSlice");
    let call = insts.iter().position(|inst| matches!(inst.op(), Op::FunctionCall { .. })).expect("the call");
    assert!(view < call);
    let Op::FunctionCall { args, .. } = insts[call].op() else { unreachable!("found as a call above") };
    assert_eq!(Some(args[0]), insts[view].result());
}

// -- the language's own operations (D79) ---------------------------------------

/// A shift's amount keeps its own width: `u8 << i64` is one `Shl`, and
/// the amount is never cast (operators §2).
#[test]
fn a_shift_amount_keeps_its_own_width() {
    let lowered = Fixture::new().script("1u8 << 3");
    let body = lowered.entry();
    let shift = body
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .find_map(|inst| match inst.op() {
            Op::BinOp { op: crate::ir::BinOp::Shl(_), left, right } => Some((*left, *right)),
            _ => None,
        });
    let Some((value, amount)) = shift else { panic!("a Shl: {}", shown(body)) };
    assert!(matches!(lowered.kind_of(body, value), TyKind::Int(IntTy::U8)));
    assert!(matches!(lowered.kind_of(body, amount), TyKind::Int(IntTy::I64)));
    assert_eq!(count(&lowered.entry_ops(), |op| matches!(op, Op::Cast { .. })), 0);
}

/// A list pattern on a slice tests the slice's length, the language's own,
/// before it reaches an element (patterns §1).
#[test]
fn a_list_pattern_on_a_slice_tests_its_length() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let slice = fx.ty(TyKind::Slice(int));
    let shared_slice = fx.reference(Mutability::Shared, slice);
    fx.input("xs", shared_slice);
    let lowered = fx.script("match $xs { [a, b] => *a, _ => 0, }");
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let length = insts.iter().position(|inst| matches!(inst.op(), Op::SliceLen { .. })).expect("a SliceLen");
    let compared = insts
        .iter()
        .position(|inst| matches!(inst.op(), Op::BinOp { op: crate::ir::BinOp::Eq, left, .. } if Some(*left) == insts[length].result()))
        .expect("the length compared with the two elements named");
    let element = insts.iter().position(|inst| matches!(inst.op(), Op::Index { .. })).expect("an element reached by index");
    assert!(compared < element, "the length is tested before an element is reached: {}", shown(body));
}

/// A declared conversion step is a call of its `via` at the function type
/// typeck recorded for it (types §7, D75): `i64` stored into a `Meters`
/// context goes through `meters(i64) -> Meters`.
#[test]
fn a_declared_conversion_step_is_a_call_of_its_via() {
    let mut fx = Fixture::new();
    let meters_id = QualifiedRef::root(fx.name("Meters"));
    fx.raw.user_types.push(UserTypeDecl {
        id: meters_id,
        type_params: Box::new([]),
        specializes: Box::new([]),
        effect_params: 0,
        identity_params: 0,
        lifetimes: 0,
    });
    let meters = fx.ty(TyKind::UserDefined {
        id: meters_id,
        type_args: Box::new([]),
        effect_args: Box::new([]),
        identity_args: Box::new([]),
    });
    let int = fx.int();
    let via_ty = fx.function(&[int], meters, Effect::pure());
    let via = fx.declare("meters", via_ty, 1);
    fx.raw.conversions.push(ConversionRule { from: int, to: meters, via });
    let m = fx.context("m", meters, false);
    let lowered = fx.script("@m = 1i64; 0");
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let call = insts
        .iter()
        .find(|inst| matches!(inst.op(), Op::FunctionCall { callee: Callee::Extern { id, .. }, .. } if *id == via))
        .and_then(|inst| inst.result())
        .expect("a call of the conversion's via");
    assert!(matches!(lowered.kind_of(body, call), TyKind::UserDefined { .. }));
    let slot = body.slots().iter().copied().find(|slot| body.context_of_slot(*slot) == Some(m)).expect("m's slot");
    assert!(insts.iter().any(|inst| matches!(
        inst.op(),
        Op::Assign { target: crate::ir::RefTarget::Var(target), value, .. } if *target == slot && *value == call
    )));
}

// -- D79: comparisons by parts, inputs in lambdas, shared captures of `&mut` ---

fn calls_of(lowered: &Lowered, callee: QualifiedRef) -> usize {
    count(&lowered.entry_ops(), |op| matches!(op, Op::FunctionCall { callee: Callee::Extern { id, .. }, .. } if *id == callee))
}

/// `==` on a tuple compares its parts in order through each part's
/// `core::eq`, the second only where the first is equal (operators §4).
#[test]
fn tuple_equality_compares_part_by_part() {
    let mut fx = Fixture::new();
    fx.comparisons();
    let lowered = fx.script("(1, true) == (2, false)");
    let eq = QualifiedRef::qualified(lowered.name("core"), lowered.name("eq"));
    assert_eq!(calls_of(&lowered, eq), 2);
    assert!(lowered.entry_terminators().iter().any(|terminator| matches!(terminator, Terminator::Diamond { .. })));
}

/// An ordering on a tuple is the first part's `cmp` that is not `0`,
/// compared with `0` (operators §2, §4).
#[test]
fn tuple_ordering_is_the_first_unequal_parts_cmp() {
    let mut fx = Fixture::new();
    fx.comparisons();
    let lowered = fx.script("(1, 2) < (1, 3)");
    let cmp = QualifiedRef::qualified(lowered.name("core"), lowered.name("cmp"));
    assert_eq!(calls_of(&lowered, cmp), 2);
    let lt = count(&lowered.entry_ops(), |op| matches!(op, Op::BinOp { op: crate::ir::BinOp::Lt, .. }));
    assert_eq!(lt, 1);
}

/// `==` on an `Option` tests both tags, and compares the payloads only
/// where both are `Some`.
#[test]
fn option_equality_tests_tags_before_payloads() {
    let mut fx = Fixture::new();
    fx.comparisons();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int));
    fx.input("o", option);
    fx.input("p", option);
    let lowered = fx.script("$o == $p");
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::TestVariant { .. })), 2);
    let eq = QualifiedRef::qualified(lowered.name("core"), lowered.name("eq"));
    assert_eq!(calls_of(&lowered, eq), 1);
}

/// An input read inside a lambda is captured as a binding is (D79).
#[test]
fn an_input_read_in_a_lambda_is_captured() {
    let mut fx = Fixture::new();
    let int = fx.int();
    fx.input("n", int);
    let lowered = fx.script("let g = |x| -> $n; g(1)");
    let closure = lowered
        .entry_ops()
        .into_iter()
        .find_map(|op| match op {
            Op::MakeClosure { body, captures } => Some((*body, captures.len())),
            _ => None,
        })
        .expect("a closure");
    assert_eq!(closure.1, 1);
    assert_eq!(lowered.module.body(closure.0).captures().len(), 1);
    assert!(lowered.module.body(closure.0).params().len() == 1, "the input is no parameter of the lambda");
}

/// A `&mut` captured by shared reference is read inside as a shared
/// reborrow, and its capture is one (functions §2, types §4).
#[test]
fn a_mut_reference_captured_shared_is_a_shared_reborrow_inside() {
    let lowered = Fixture::new().script("let a = 1; let r = &mut a; let g = |x| -> *r; g(1)");
    let entry = lowered.entry();
    let captures = lowered
        .entry_ops()
        .into_iter()
        .find_map(|op| match op {
            Op::MakeClosure { captures, .. } => Some(captures.clone()),
            _ => None,
        })
        .expect("a closure");
    assert!(matches!(lowered.kind_of(entry, captures[0]), TyKind::Ref(Mutability::Shared, _)));
}

// -- D93: a function type's holds and calls; gap 7 ----------------------------

/// Parsed, checked, lowered, and its entry's loans solved.
fn lowered_with_loans(fx: Fixture, source: &str) -> (Lowered, std::rc::Rc<crate::passes::analysis::Loans>) {
    let script: Script = acvus_ast::parse_script(&fx.names, source).expect("the test's source parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let mut types = Types::new();
    let typeck = check_script(&decls, &mut types, &script);
    assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
    let module = lower_script(&decls, &typeck, types, &script);
    let mut am = crate::passes::analysis::AnalysisManager::new(&decls);
    let loans = am.get::<crate::passes::analysis::Loans>(&module, module.entry());
    (Lowered::checked(module, fx.names), loans)
}

/// ownership §5, functions §2, D93: `|x| -> v.push(x)` lends `v` `&mut`, so
/// the lambda holds `Mut`, a fact settled after the fixpoint. The slot `f`
/// lives in has the closure's own type, and the closure and the callee
/// read out of `f` both hold the exclusive loan on `v`'s slot that the
/// borrow check reads.
#[test]
fn a_lambda_pushing_to_a_captured_vec_holds_it_mut_in_its_slot_and_its_closure() {
    let mut fx = Fixture::new();
    let vec_ty = fx.vec_of_ints();
    let int = fx.int();
    let unit = fx.ty(TyKind::Unit);
    let vec_fn = fx.function(&[], vec_ty, Effect::pure());
    fx.declare("vec", vec_fn, 0);
    let lent_vec = fx.reference(Mutability::Mut, vec_ty);
    let push_fn = fx.function(&[lent_vec, int], unit, Effect::pure());
    fx.declare("push", push_fn, 2);
    let (lowered, loans) = lowered_with_loans(fx, "let v = vec(); let f = |x| -> v.push(x); f(1)");
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let (closure, captures) = insts
        .iter()
        .find_map(|inst| match inst.op() {
            Op::MakeClosure { captures, .. } => Some((inst.result().expect("a closure is a value"), captures.clone())),
            _ => None,
        })
        .unwrap_or_else(|| panic!("a closure: {}", shown(body)));
    let TyKind::Fn { calls, holds, .. } = lowered.kind_of(body, closure) else {
        panic!("a closure is a function value")
    };
    assert_eq!((*calls, *holds), (Calls::Many, Holds::Mut));
    let slot = insts
        .iter()
        .find_map(|inst| match inst.op() {
            Op::Assign { target: RefTarget::Var(slot), path, value, .. } if path.is_empty() && *value == closure => Some(*slot),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the closure is put in `f`'s slot: {}", shown(body)));
    assert_eq!(body.ty_of(slot), body.ty_of(closure), "`f`'s slot has the closure's type");

    let [capture] = captures[..] else { panic!("the closure captures `v` alone: {}", shown(body)) };
    let v = insts
        .iter()
        .find_map(|inst| match inst.op() {
            Op::Ref { target: RefTarget::Var(v), path, mutability: Mutability::Mut } if path.is_empty() && inst.result() == Some(capture) => {
                Some(*v)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("`v` is captured by `&mut`: {}", shown(body)));
    let on_v = Loan { storage: Storage::Slot(v), mode: Mode::Exclusive };
    assert!(loans.of(closure, 0).contains(&on_v), "the closure holds `v` exclusively: {:?}", loans.of(closure, 0));
    let callee = insts
        .iter()
        .find_map(|inst| match inst.op() {
            Op::FunctionCall { callee: Callee::Indirect(callee), .. } => Some(*callee),
            _ => None,
        })
        .unwrap_or_else(|| panic!("`f(1)` calls a function value: {}", shown(body)));
    assert_eq!(body.ty_of(callee), body.ty_of(closure), "the callee read out of `f` has the closure's type");
    assert!(loans.of(callee, 0).contains(&on_v), "the callee holds `v` exclusively: {:?}", loans.of(callee, 0));
}

/// ownership §5, gap 7: a `&mut` part of a held scrutinee does not copy,
/// and each use reborrows it. Every test and bind that goes through it
/// reads it through a `&mut` to the scrutinee's slot, so no part of the
/// slot is taken for it. After drop insertion, each arm takes the string
/// part once, for its bind, and ends the `&mut` part once, where the arm
/// no longer reads it; the pair is never taken or dropped whole.
#[test]
fn a_match_through_a_held_mut_reference_reborrows_it_and_moves_no_part() {
    let mut fx = Fixture::new();
    fx.comparisons();
    let string = fx.ty(TyKind::String);
    let name_fn = fx.function(&[], string, Effect::pure());
    fx.declare("name", name_fn, 0);
    let script: Script = acvus_ast::parse_script(
        &fx.names,
        "let p = (1, 2); match (&mut p, name()) { ((0, b), t) => { *b = 5; t }, ((a, _), t) => { *a = 3; t }, }",
    )
    .expect("parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let mut types = Types::new();
    let typeck = check_script(&decls, &mut types, &script);
    assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
    let mut module = lower_script(&decls, &typeck, types, &script);
    let mut am = crate::passes::analysis::AnalysisManager::new(&decls);
    let before: Vec<Op> = module.body(module.entry()).blocks().flat_map(|(_, block)| block.insts()).map(|inst| inst.op().clone()).collect();
    super::required::insert_drops(&mut module, &mut am);
    let lowered = Lowered::checked(module, fx.names);
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let slot = insts
        .iter()
        .find_map(|inst| match inst.op() {
            Op::Assign { target: RefTarget::Var(slot), path, .. }
                if path.is_empty()
                    && matches!(lowered.kind_of(body, *slot), TyKind::Tuple(items)
                        if matches!(lowered.module.types().kind(items[0]), TyKind::Ref(Mutability::Mut, _))) =>
            {
                Some(*slot)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("the pair is put in a slot: {}", shown(body)));
    let takes = |ops: &[&Op], wanted: &[PathSeg]| {
        count(ops, |op| matches!(op, Op::Take { target: RefTarget::Var(named), path, .. } if *named == slot && path == wanted))
    };
    let before: Vec<&Op> = before.iter().collect();
    assert_eq!(takes(&before, &[PathSeg::Index(0)]), 0, "lowering takes the `&mut` part nowhere: {}", shown(body));

    let lent_slot: Vec<_> = insts
        .iter()
        .filter(|inst| matches!(inst.op(), Op::Ref { target: RefTarget::Var(named), path, mutability: Mutability::Mut } if *named == slot && path.is_empty()))
        .filter_map(|inst| inst.result())
        .collect();
    let reborrowed = count(&lowered.entry_ops(), |op| {
        matches!(op, Op::Take { target: RefTarget::Through(lent), path, .. } if lent_slot.contains(lent) && *path == [PathSeg::Index(0)])
    });
    assert_eq!(reborrowed, 3, "the test and each arm's bind read the `&mut` part through a `&mut` to the slot: {}", shown(body));

    for (arm, block) in body.blocks().filter(|(_, block)| block.insts().any(|inst| matches!(inst.op(), Op::Const { value: acvus_ast::Literal::Int(5 | 3) }))) {
        let ops: Vec<&Op> = block.insts().map(|inst| inst.op()).collect();
        assert_eq!(takes(&ops, &[]), 0, "arm {arm:?} never takes the pair whole: {}", shown(body));
        assert_eq!(takes(&ops, &[PathSeg::Index(1)]), 1, "arm {arm:?} takes the string once, for `t`: {}", shown(body));
        let ended: Vec<_> = block
            .insts()
            .filter(|inst| matches!(inst.op(), Op::Take { target: RefTarget::Var(named), path, .. } if *named == slot && *path == [PathSeg::Index(0)]))
            .filter_map(|inst| inst.result())
            .collect();
        let [ended] = ended[..] else { panic!("arm {arm:?} ends the `&mut` part once: {}", shown(body)) };
        assert_eq!(count(&ops, |op| matches!(op, Op::Drop { src } if *src == ended)), 1, "and drops what it took: {}", shown(body));
    }
    assert!(!lowered.entry_ops().iter().any(|op| matches!(op, Op::Drop { src } if *src == slot)), "the pair is never dropped whole: {}", shown(body));
}

// -- D87: part trees; arrays in one loop; no structural instruction ---------

/// `clone(&String) -> String`, pure.
fn clones_strings(fx: &mut Fixture) {
    let string = fx.ty(TyKind::String);
    let shared = fx.reference(Mutability::Shared, string);
    let clones = fx.function(&[shared], string, Effect::pure());
    let clone = fx.raw.core.clone;
    fx.lang_item(clone, &[clones]);
}

fn no_structural_instruction(lowered: &Lowered) {
    for body in lowered.module.body_ids() {
        let structural = count(&lowered.ops(body), |op| matches!(op, Op::StructuralEq { .. } | Op::StructuralClone { .. }));
        assert_eq!(structural, 0, "D87: {}", shown(lowered.module.body(body)));
    }
}

fn fors(lowered: &Lowered) -> Vec<(&ForSource, crate::ir::cfg::BlockId, crate::ir::cfg::BlockId)> {
    lowered
        .entry_terminators()
        .into_iter()
        .filter_map(|terminator| match terminator {
            Terminator::For { source, stages, exit, .. } => Some((source, stages.body(), *exit)),
            _ => None,
        })
        .collect()
}

/// operators §4, D87: `==` on arrays is one `For` over the indices, not
/// one comparison per element; the loop leaves for its exit at the first
/// element that is not equal, with that answer, which the exit takes.
#[test]
fn array_equality_is_one_for_that_exits_at_the_first_difference() {
    let mut fx = Fixture::new();
    fx.comparisons();
    let int = fx.int();
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(3)));
    fx.input("a", array);
    fx.input("b", array);
    let lowered = fx.script("$a == $b");
    let body = lowered.entry();
    let [(source, _, exit)] = fors(&lowered)[..] else { panic!("one For: {}", shown(body)) };
    assert!(matches!(source, ForSource::Range { .. }), "{}", shown(body));
    let [answer] = body.block(exit).params() else { panic!("the exit takes the answer: {}", shown(body)) };
    assert_eq!(lowered.kind_of(body, *answer), &TyKind::Bool);
    let exits_early = lowered.entry_terminators().into_iter().any(|terminator| {
        matches!(terminator, Terminator::JumpIf { else_to, else_args, .. } if *else_to == exit && else_args.len() == 1)
    });
    assert!(exits_early, "{}", shown(body));
    let eq = QualifiedRef::qualified(lowered.name("core"), lowered.name("eq"));
    assert_eq!(calls_of(&lowered, eq), 1, "one element comparison, in the loop: {}", shown(body));
    no_structural_instruction(&lowered);
}

/// operators §2, §4, D87: an ordering on arrays is one `For` that leaves
/// at the first `cmp` that is not `0`, and the answer is compared with `0`
/// after it.
#[test]
fn array_ordering_is_one_for_that_exits_at_the_first_nonzero_cmp() {
    let mut fx = Fixture::new();
    fx.comparisons();
    let int = fx.int();
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(4)));
    fx.input("a", array);
    fx.input("b", array);
    let lowered = fx.script("$a < $b");
    let body = lowered.entry();
    let [(_, _, exit)] = fors(&lowered)[..] else { panic!("one For: {}", shown(body)) };
    let [answer] = body.block(exit).params() else { panic!("the exit takes the answer: {}", shown(body)) };
    assert_eq!(lowered.kind_of(body, *answer), &TyKind::Int(IntTy::I64));
    let cmp = QualifiedRef::qualified(lowered.name("core"), lowered.name("cmp"));
    assert_eq!(calls_of(&lowered, cmp), 1, "{}", shown(body));
    let lt = count(&lowered.entry_ops(), |op| matches!(op, Op::BinOp { op: crate::ir::BinOp::Lt, left, .. } if left == answer));
    assert_eq!(lt, 1, "{}", shown(body));
}

/// ownership §1-2, D87: `.clone()` of a tuple of a `String` and an `i64`
/// builds a new tuple: the `String` through its `core::clone` impl, the
/// `i64` copied out of the receiver.
#[test]
fn a_tuple_clone_is_built_part_by_part() {
    let mut fx = Fixture::new();
    clones_strings(&mut fx);
    let string = fx.ty(TyKind::String);
    let int = fx.int();
    let pair = fx.ty(TyKind::Tuple(Box::new([string, int])));
    fx.input("t", pair);
    let lowered = fx.script("$t.clone()");
    let body = lowered.entry();
    let clone = QualifiedRef::qualified(lowered.name("core"), lowered.name("clone"));
    assert_eq!(calls_of(&lowered, clone), 1, "{}", shown(body));
    let copied_int = count(&lowered.entry_ops(), |op| {
        matches!(op, Op::Take { target: RefTarget::Through(_), path, .. } if path.is_empty())
    });
    assert_eq!(copied_int, 1, "{}", shown(body));
    assert_eq!(count(&lowered.entry_ops(), |op| matches!(op, Op::MakeTuple { elements } if elements.len() == 2)), 1, "{}", shown(body));
    no_structural_instruction(&lowered);
}

/// ownership §1, D87: an array whose element copies is copied whole by
/// `.clone()`: no loop and no call.
#[test]
fn an_array_that_copies_is_copied_by_clone() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(3)));
    fx.input("a", array);
    let lowered = fx.script("$a.clone()");
    let body = lowered.entry();
    assert!(fors(&lowered).is_empty(), "{}", shown(body));
    assert_eq!(count(&lowered.entry_ops(), |op| matches!(op, Op::FunctionCall { .. })), 0, "{}", shown(body));
    no_structural_instruction(&lowered);
}

/// D87: no comparison or clone of a structural type lowers to a structural
/// instruction; `Option` and `Result` clones test the tag and rebuild the
/// variant.
#[test]
fn no_structural_instruction_is_lowered() {
    let mut fx = Fixture::new();
    fx.comparisons();
    clones_strings(&mut fx);
    let string = fx.ty(TyKind::String);
    let int = fx.int();
    let option = fx.ty(TyKind::Option(string));
    let result = fx.ty(TyKind::Result(string, int));
    let a = fx.name("a");
    let object = fx.ty(TyKind::Object(crate::ty::ObjectTy::Structural(Box::new([(a, int)]), ())));
    fx.input("o", option);
    fx.input("r", result);
    fx.input("x", object);
    let lowered = fx.script("let p = $o.clone(); let q = $r.clone(); let e = $x == $x; let y = $x.clone(); e");
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::MakeVariant { .. })), 4, "{}", shown(lowered.entry()));
    no_structural_instruction(&lowered);
}

/// D87: an array comparison's loop and a tuple clone go through the
/// required steps and the full pipeline, every body keeping the editor's
/// invariants.
#[test]
fn part_trees_survive_the_required_steps_and_the_pipeline() {
    let mut fx = Fixture::new();
    fx.comparisons();
    clones_strings(&mut fx);
    let string = fx.ty(TyKind::String);
    let int = fx.int();
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(3)));
    let pair = fx.ty(TyKind::Tuple(Box::new([string, int])));
    fx.input("a", array);
    fx.input("b", array);
    fx.input("t", pair);
    let script: Script = acvus_ast::parse_script(&fx.names, "let c = $t.clone(); $a < $b").expect("parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let mut types = Types::new();
    let typeck = check_script(&decls, &mut types, &script);
    assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
    let mut module = lower_script(&decls, &typeck, types, &script);
    let mut am = crate::passes::analysis::AnalysisManager::new(&decls);
    super::required::clone_strings(&mut module, &mut am);
    super::required::remove_dead_context_ops(&mut module, &mut am);
    super::required::insert_drops(&mut module, &mut am);
    crate::passes::pipeline::optimize(&mut module, &mut am, crate::passes::pipeline::Level::Full);
    let lowered = Lowered::checked(module, fx.names);
    no_structural_instruction(&lowered);
}
