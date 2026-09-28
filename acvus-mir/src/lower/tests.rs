//! Each form, from source to IR: parsed with `acvus-ast`, checked by
//! typeck over declarations that went through `verify`, lowered, and every
//! body held to the editor's invariants.

use acvus_ast::Script;
use acvus_utils::{Astr, Interner, QualifiedRef};

use super::{lower_script, lower_template};
use crate::decl::{
    ContextDecl, ConversionRule, Core, Declarations, ImplSig, Impls, LoopMarkers, RawDeclarations, Scheme, TyVarBound, UserTypeDecl,
};
use crate::ir::cfg::edit::{DefDominatesUses, Invariant, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree, Violation};
use crate::ir::cfg::{Body, BodyId, Module, Op, OrderUse, Terminator};
use crate::ir::{Callee, ForSource, SwitchKey};
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
        let scheme = Scheme { ty, param_names: self.param_names(arity), bounds: Box::new([]), impls: None, requires: Box::new([]) };
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

    /// `display(&i64, &mut String)` and `display(&str, &mut String)`, pure:
    /// typeck reaches `display` for a template's text too.
    fn display_ints(&mut self) {
        let int = self.int();
        let shown = self.reference(Mutability::Shared, int);
        let string = self.ty(TyKind::String);
        let out = self.reference(Mutability::Mut, string);
        let unit = self.ty(TyKind::Unit);
        let impl_ty = self.function(&[shown, out], unit, Effect::pure());
        let str_ty = self.ty(TyKind::Str);
        let shown_text = self.reference(Mutability::Shared, str_ty);
        let text_ty = self.function(&[shown_text, out], unit, Effect::pure());
        let display = self.raw.core.display;
        self.lang_item(display, &[impl_ty, text_ty]);
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
    let else_ops = lowered.entry().block(diamond).insts();
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
    let bound = body.block(stage).insts();
    assert!(matches!(bound[0].op(), Op::UnwrapVariant { .. }), "the body binds x first: {}", shown(body));
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
    assert!(count(&ops, |op| matches!(op, Op::StringAppend { .. })) >= 1);
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
