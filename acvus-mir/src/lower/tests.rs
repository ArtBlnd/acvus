//! Each form, from source to IR: parsed with `acvus-ast`, checked by
//! typeck over declarations that went through `verify`, lowered, and every
//! body held to the editor's invariants.

use acvus_ast::Script;
use acvus_utils::{Astr, Interner, Namespace, QualifiedRef};

use super::{lower_script, lower_template};
use crate::decl::{
    ContextDecl, ConversionRule, Core, Declarations, DeclaredFacts, ImplSig, Impls, LoopMarkers, RawDeclarations, Scheme, TyVarBound, UserTypeDecl,
};
use crate::ir::cfg::edit::{DefDominatesUses, Invariant, NeverDiverges, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree, Violation};
use crate::ir::cfg::{Body, BodyId, Module, Op, OrderUse, Terminator};
use crate::ir::{Callee, ForSource, PathSeg, RefTarget, SwitchKey};
use crate::passes::analysis::{Loan, Mode, Storage};
use crate::ty::{Calls, Concrete, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Poly, TyId, TyKind, TypeArg, Types};
use crate::typeck::{Inputs, Interfaces, check_script, check_template};

struct Fixture {
    names: Interner,
    raw: RawDeclarations,
    /// Each input `$x` with its type, at which a run's inputs are fixed.
    inputs: Vec<(Astr, TyId<Poly>)>,
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
                entries: Vec::new(),
                loop_markers: LoopMarkers::default(),
                core,
            },
            inputs: Vec::new(),
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
            values: (),
        })
    }

    fn param_names(&self, count: usize) -> Box<[Astr]> {
        (0..count).map(|at| self.name(&format!("p{at}"))).collect()
    }

    /// A plain extern fn `name` of type `ty`, with no type variable.
    fn declare(&mut self, name: &str, ty: TyId<Poly>, arity: usize) -> QualifiedRef {
        let id = QualifiedRef::root(self.name(name));
        let scheme = Scheme { ty, param_names: self.param_names(arity), bounds: Box::new([]), impls: None, requires: Box::new([]), facts: DeclaredFacts::default(), runs: Box::new([]) };
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
        self.inputs.push((name, ty));
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
                bounds: vec![TyVarBound::Any; variables_of(&self.raw.types, *ty)].into(),
                param_names: self.param_names(match self.raw.types.kind(*ty) {
                    TyKind::Fn { params, .. } => params.len(),
                    _ => 0,
                }),
                requires: Box::new([]),
                facts: Default::default(),
                runs: Box::new([]),
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
            carries_effect: false,
            carries_identity: false,
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
            effect: None,
            identity: None,
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
        self.script_calling(&Interfaces::new(), source)
    }

    /// `source` lowered and taken through every required step, which admit
    /// it.
    fn required(self, source: &str) -> Lowered {
        let script: Script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_script(&self.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &script);
        assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
        let mut module = lower_script(&decls, &typeck, types, &script);
        let mut am = crate::passes::analysis::AnalysisManager::new(&decls);
        if let Err(refusals) = super::required::run(&mut module, &mut am) {
            panic!("the checks refused: {refusals:?}");
        }
        Lowered::checked(module, self.names)
    }

    /// `source` lowered, its calls reaching the units of `interfaces` too.
    fn script_calling(self, interfaces: &Interfaces, source: &str) -> Lowered {
        let script: Script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_script(&self.names, &decls, interfaces, Inputs::Fixed(&inputs), &mut types, &script);
        assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
        let module = lower_script(&decls, &typeck, types, &script);
        Lowered::checked(module, self.names)
    }

    /// `source` lowered, its calls reaching `units`, each a script at
    /// `namespace::name` checked with its inputs open against the same
    /// declarations first.
    fn script_with_units(self, units: &[(&[&str], &str, &str)], source: &str) -> (Lowered, Vec<QualifiedRef>) {
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut interfaces = Interfaces::new();
        let mut names = Vec::new();
        for (namespace, name, unit_source) in units {
            let unit_script: Script = acvus_ast::parse_script(&self.names, unit_source).expect("the unit's source parses");
            let checked = check_script(&self.names, &decls, &interfaces, Inputs::Open, &mut Types::new(), &unit_script);
            assert!(checked.refusals.is_empty(), "the unit {name} is refused: {:?}", checked.refusals);
            let path: Vec<Astr> = namespace.iter().map(|segment| self.names.intern(segment)).collect();
            let unit = QualifiedRef::qualified(self.names.namespace(&path), self.names.intern(name));
            interfaces.set(unit, checked.interface.expect("an open check gives its interface"));
            names.push(unit);
        }
        let script: Script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_script(&self.names, &decls, &interfaces, Inputs::Fixed(&inputs), &mut types, &script);
        assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
        let module = lower_script(&decls, &typeck, types, &script);
        (Lowered::checked(module, self.names), names)
    }

    fn template(self, source: &str) -> Lowered {
        let template = acvus_ast::parse_template(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_template(&self.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &template);
        assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
        let module = lower_template(&decls, &typeck, types, &template);
        Lowered::checked(module, self.names)
    }
}

/// How many type variables `ty` holds: one past the largest index.
fn variables_of(types: &Types<Poly>, ty: TyId<Poly>) -> usize {
    let mut largest: Option<u32> = None;
    let mut stack = vec![ty];
    while let Some(at) = stack.pop() {
        let kind = types.kind(at);
        if let TyKind::Var(var) = kind {
            largest = Some(largest.map_or(*var, |known| known.max(*var)));
        }
        kind.for_each_child(|child| stack.push(child));
    }
    largest.map_or(0, |largest| largest as usize + 1)
}

/// A lowered module whose every body holds the editor's invariants.
struct Lowered {
    module: Module,
    names: Interner,
}

type Checker = fn(&Body, &Types<Concrete>) -> Result<(), Violation>;

impl Lowered {
    fn checked(module: Module, names: Interner) -> Lowered {
        let checkers: [Checker; 6] = [
            <NeverDiverges as Invariant>::check,
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

    fn namespace(&self, path: &[&str]) -> Namespace {
        let path: Vec<Astr> = path.iter().map(|segment| self.names.intern(segment)).collect();
        self.names.namespace(&path)
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

/// A `match` over one enum's tags is one `Switch` holding its join (D57),
/// its missing keys defaulting to the block no arm answers.
#[test]
fn a_match_over_tags_is_one_switch() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int, ()));
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
    // No catch-all: a value no key names goes to the block where no arm
    // answers, which matchck judges (patterns §3).
    let (unmatched, _) = default.as_ref().expect("a Switch with no catch-all defaults to the block no arm answers");
    assert!(matches!(lowered.entry().block(*unmatched).terminator(), Terminator::NoArmMatched { .. }));
    assert_eq!(lowered.entry().block(*join).params().len(), 1);
    assert_eq!(count(&lowered.entry_ops(), |op| matches!(op, Op::TestVariant { .. })), 0);
}

/// A `match` whose arm tests a nested pattern is tested arm by arm, in
/// order, each arm a `Diamond` to the one join (D57).
#[test]
fn a_nested_pattern_is_tested_arm_by_arm() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int, ()));
    let pair = fx.ty(TyKind::Tuple(Box::new([option, int]), ()));
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
    let option = fx.ty(TyKind::Option(int, ()));
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
    let option = fx.ty(TyKind::Option(string, ()));
    let pair = fx.ty(TyKind::Tuple(Box::new([option, string]), ()));
    let pair_fn = fx.function(&[], pair, Effect::pure());
    fx.declare("pair", pair_fn, 0);
    let script: Script = acvus_ast::parse_script(&fx.names, "if let (Some(a), b) = pair() { 1 } else { 2 }").expect("parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let mut types = Types::new();
    let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &fx.inputs);
    let typeck = check_script(&fx.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &script);
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
                if path.is_empty() && matches!(lowered.kind_of(body, *slot), TyKind::Tuple(items, _) if items.len() == 2) =>
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
    let option = fx.ty(TyKind::Option(int, ()));
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

/// programs §1, §2, graph: a call of a unit is a call of its instantiation
/// at the call's types, the unit named directly; its arguments are passed
/// in the order of its parameters, its inputs' names' order (`times` before
/// `user`), whatever order they are written in.
#[test]
fn a_unit_call_passes_its_arguments_in_the_parameters_order() {
    let mut fx = Fixture::new();
    let (bool_ty, int, string) = (fx.ty(TyKind::Bool), fx.int(), fx.ty(TyKind::String));
    let both = fx.function(&[bool_ty, int], string, Effect::pure());
    fx.declare("both", both, 2);
    let (lowered, units) = fx.script_with_units(&[(&["ns"], "greet", "both($user, $times)")], "ns::greet(user: true, times: 2)");
    let greet = units[0];
    let body = lowered.entry();
    let passed = lowered
        .entry_ops()
        .into_iter()
        .find_map(|op| match op {
            Op::FunctionCall { callee: Callee::Direct(unit), args, .. } if *unit == greet => Some(args.clone()),
            _ => None,
        })
        .expect("the unit is called directly");
    let kinds: Vec<&TyKind<Concrete>> = passed.iter().map(|arg| lowered.kind_of(body, *arg)).collect();
    assert_eq!(kinds, vec![&TyKind::Int(IntTy::I64), &TyKind::Bool]);
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
    let mut fx = Fixture::new();
    let int = fx.int();
    let shared = fx.reference(Mutability::Shared, int);
    let adds = fx.function(&[shared, shared], int, Effect::pure());
    let add = fx.raw.core.add;
    fx.lang_item(add, &[adds]);
    let lowered = fx.script("let a = 1; let g = |x| -> a; let b = 2; let h = move |x| -> b; g(1) + 0; h(1)");
    lowered_closures(&lowered);
}

fn lowered_closures(lowered: &Lowered) {
    let closures: Vec<(BodyId, Vec<crate::ir::ValueId>)> = lowered
        .entry_ops()
        .into_iter()
        .filter_map(|op| match op {
            Op::MakeClosure { bodies, captures } => Some((bodies[0], captures.clone())),
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

/// Whether `op` is a place operation on `context`: its `Take`, its `Ref`
/// or its `Assign`, named by `kind`.
fn names_context(op: &Op, context: QualifiedRef) -> Option<&'static str> {
    let kind = match op {
        Op::Take { .. } => "take",
        Op::Ref { .. } => "lend",
        Op::Assign { .. } => "assign",
        _ => return None,
    };
    matches!(op.place(), Some((RefTarget::Context(named), _)) if *named == context).then_some(kind)
}

/// A context is a place by name (context §3): read by a take and written
/// by an assign where the program names it, with nothing around the call
/// that may touch it, nothing at entry and nothing at the return. The body
/// states it among its contexts, and a context not volatile stands off the
/// order chain.
#[test]
fn a_context_is_a_place_by_name_with_nothing_at_entry_around_a_call_or_at_return() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let n = fx.context("n", int, false);
    fx.opaque_f();
    let lowered = fx.script("let a = @n; f(1); @n = a; 1");
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let shape: Vec<&str> = insts
        .iter()
        .filter_map(|inst| match inst.op() {
            Op::FunctionCall { .. } => Some("call"),
            op => names_context(op, n),
        })
        .collect();
    assert_eq!(shape, vec!["take", "call", "assign"], "{}", shown(body));
    assert_eq!(body.contexts().iter().map(|named| named.context).collect::<Vec<_>>(), vec![n]);
    let off_the_chain = insts.iter().filter(|inst| names_context(inst.op(), n).is_some()).all(|inst| inst.order().is_none());
    assert!(off_the_chain, "a context not volatile carries no token: {}", shown(body));
}

/// A volatile context is a place as any other, and each of its place
/// operations is an effect on the order chain (context §3).
#[test]
fn a_volatile_contexts_place_operations_carry_order_tokens() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let clock = fx.context("clock", int, true);
    let lowered = fx.script("let a = @clock; let b = @clock; @clock = a; 1");
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let on_clock: Vec<_> = insts.iter().filter_map(|inst| names_context(inst.op(), clock).map(|kind| (kind, inst.order().is_some()))).collect();
    assert_eq!(on_clock, vec![("take", true), ("take", true), ("assign", true)], "{}", shown(body));
}

/// context §3: after the required steps a context is lent and assigned
/// where the program does, and an overwrite of a content the body holds is
/// preceded by a take of that content, which is dropped.
#[test]
fn a_context_is_lent_and_assigned_as_written_through_the_required_steps() {
    let shape = |source: &str| {
        let mut fx = Fixture::new();
        let vec = fx.vec_of_ints();
        let (int, vec_ref) = (fx.int(), fx.reference(Mutability::Shared, vec));
        let len = fx.function(&[vec_ref], int, Effect::pure());
        fx.declare("len", len, 1);
        let make = fx.function(&[], vec, Effect::pure());
        fx.declare("make", make, 0);
        let c = fx.context("c", vec, false);
        let lowered = fx.required(source);
        let ops = lowered.entry_ops();
        ops.iter().filter_map(|op| names_context(op, c)).collect::<Vec<_>>()
    };
    assert_eq!(shape("len(&@c)"), vec!["lend"], "only read");
    assert_eq!(shape("@c = make(); len(&@c)"), vec!["take", "assign", "lend"], "written before any read");
    assert_eq!(shape("let n = len(&@c); @c = make(); n"), vec!["lend", "take", "assign"], "read, then written");
}

/// context §1, §3: a volatile `String` context read twice is admitted
/// through the required steps: each read, its content read again by the
/// return, becomes a clone of a shared lend on the order chain.
#[test]
fn a_volatile_string_context_read_twice_is_admitted_through_clones() {
    let mut fx = Fixture::new();
    let (int, string) = (fx.int(), fx.ty(TyKind::String));
    let both = fx.function(&[string, string], int, Effect::pure());
    fx.declare("both", both, 2);
    let tape = fx.context("tape", string, true);
    let lowered = fx.required("let a = @tape; let b = @tape; both(a, b)");
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let on_tape: Vec<_> = insts.iter().filter_map(|inst| names_context(inst.op(), tape).map(|kind| (kind, inst.order().is_some()))).collect();
    assert_eq!(on_tape, vec![("lend", true), ("lend", true)], "{}", shown(body));
}

/// context §3: the full pipeline promotes no context. The store before the
/// call stays, since the opaque call reads the context, and the tail takes
/// the context after the call and returns what it took.
#[test]
fn a_context_stays_a_place_through_the_pipeline() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let n = fx.context("n", int, false);
    fx.opaque_f();
    let source = "@n = 2; f(1); @n";
    let script: Script = acvus_ast::parse_script(&fx.names, source).expect("the test's source parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let mut types = Types::new();
    let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &fx.inputs);
    let typeck = check_script(&fx.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &script);
    assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
    let mut module = lower_script(&decls, &typeck, types, &script);
    let mut am = crate::passes::analysis::AnalysisManager::new(&decls);
    super::required::run(&mut module, &mut am).expect("the checks admit it");
    crate::passes::pipeline::optimize(&mut module, &mut am, crate::passes::pipeline::Level::Full);
    let lowered = Lowered::checked(module, fx.names);
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let shape: Vec<&str> = insts
        .iter()
        .filter_map(|inst| match inst.op() {
            Op::FunctionCall { .. } => Some("call"),
            op => names_context(op, n),
        })
        .collect();
    assert_eq!(shape, vec!["assign", "call", "take"], "{}", shown(body));
    let stored = insts.iter().find_map(|inst| match inst.op() {
        Op::Assign { target: RefTarget::Context(named), value, .. } if *named == n => Some(*value),
        _ => None,
    });
    let stored = stored.expect("the store stays");
    assert!(
        matches!(insts.iter().find(|inst| inst.result() == Some(stored)).map(|inst| inst.op()), Some(Op::Const { .. })),
        "the store before the call stores the constant: {}",
        shown(body)
    );
    let taken = insts.iter().find(|inst| names_context(inst.op(), n) == Some("take")).and_then(|inst| inst.result());
    let returned = body.blocks().find_map(|(_, block)| match block.terminator() {
        Terminator::Return { value, .. } => Some(*value),
        _ => None,
    });
    assert_eq!(returned, taken, "the tail returns what it took after the call: {}", shown(body));
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
    let display = QualifiedRef::qualified(lowered.namespace(&["core"]), lowered.name("display"));
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
        carries_effect: false,
        carries_identity: false,
        lifetimes: 0,
    });
    let meters = fx.ty(TyKind::UserDefined {
        id: meters_id,
        type_args: Box::new([]),
        effect: None,
        identity: None,
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
    assert!(insts.iter().any(|inst| matches!(
        inst.op(),
        Op::Assign { target: RefTarget::Context(target), value, .. } if *target == m && *value == call
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
    let eq = QualifiedRef::qualified(lowered.namespace(&["core"]), lowered.name("eq"));
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
    let cmp = QualifiedRef::qualified(lowered.namespace(&["core"]), lowered.name("cmp"));
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
    let option = fx.ty(TyKind::Option(int, ()));
    fx.input("o", option);
    fx.input("p", option);
    let lowered = fx.script("$o == $p");
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::TestVariant { .. })), 2);
    let eq = QualifiedRef::qualified(lowered.namespace(&["core"]), lowered.name("eq"));
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
            Op::MakeClosure { bodies, captures } => Some((bodies[0], captures.len())),
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
    let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &fx.inputs);
    let typeck = check_script(&fx.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &script);
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
    let on_v = |loan: &Loan| loan.storage == Storage::Slot(v) && loan.path.is_empty() && loan.mode == Mode::Exclusive;
    assert!(loans.of(closure, 0).iter().any(on_v), "the closure holds `v` exclusively: {:?}", loans.of(closure, 0));
    let callee = insts
        .iter()
        .find_map(|inst| match inst.op() {
            Op::FunctionCall { callee: Callee::Indirect(callee), .. } => Some(*callee),
            _ => None,
        })
        .unwrap_or_else(|| panic!("`f(1)` calls a function value: {}", shown(body)));
    assert_eq!(body.ty_of(callee), body.ty_of(closure), "the callee read out of `f` has the closure's type");
    assert!(loans.of(callee, 0).iter().any(on_v), "the callee holds `v` exclusively: {:?}", loans.of(callee, 0));
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
    let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &fx.inputs);
    let typeck = check_script(&fx.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &script);
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
                    && matches!(lowered.kind_of(body, *slot), TyKind::Tuple(items, _)
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

// -- D87: part trees; arrays in one loop -----------------------------------

/// `clone(&String) -> String`, pure.
fn clones_strings(fx: &mut Fixture) {
    let string = fx.ty(TyKind::String);
    let shared = fx.reference(Mutability::Shared, string);
    let clones = fx.function(&[shared], string, Effect::pure());
    let clone = fx.raw.core.clone;
    fx.lang_item(clone, &[clones]);
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
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(3), ()));
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
    let eq = QualifiedRef::qualified(lowered.namespace(&["core"]), lowered.name("eq"));
    assert_eq!(calls_of(&lowered, eq), 1, "one element comparison, in the loop: {}", shown(body));
}

/// The values a part read of a `&mut` part yields in `source`'s entry, each
/// with its loans, checked to be shared reborrows of slots lent `&mut`.
fn mut_parts_read_shared(source: &str, read: impl Fn(&Op) -> bool) -> usize {
    let mut fx = Fixture::new();
    fx.comparisons();
    let (lowered, loans) = lowered_with_loans(fx, source);
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let lent_mut: std::collections::BTreeSet<Storage> = insts
        .iter()
        .filter_map(|inst| match inst.op() {
            Op::Ref { target: RefTarget::Var(slot), path, mutability: Mutability::Mut } if path.is_empty() => Some(Storage::Slot(*slot)),
            _ => None,
        })
        .collect();
    let eq = QualifiedRef::qualified(lowered.namespace(&["core"]), lowered.name("eq"));
    assert!(calls_of(&lowered, eq) >= 1, "the `i64`s the parts name are compared: {}", shown(body));
    let reads: Vec<crate::ir::ValueId> = read_values(body, &read)
        .into_iter()
        .filter(|result| matches!(lowered.kind_of(body, *result), TyKind::Ref(..)))
        .collect();
    for at in &reads {
        assert!(matches!(lowered.kind_of(body, *at), TyKind::Ref(Mutability::Shared, _)), "read shared: {}", shown(body));
        let own = loans.of(*at, 0);
        assert!(!own.is_empty(), "the read holds a loan on what the part names: {}", shown(body));
        for loan in own {
            assert_eq!(loan.mode, Mode::Shared, "a reborrow through a shared path is shared: {own:?}");
            assert!(lent_mut.contains(&loan.storage), "the loan is on a slot lent `&mut`: {own:?}");
        }
    }
    reads.len()
}

/// ownership §5, types §4, D108: a `&mut` element of an array compared is
/// read as a shared reborrow of it, never copied out as a `&mut`, and
/// compares what it names.
#[test]
fn an_array_of_mut_references_compares_by_shared_reborrow() {
    let source = "let a = 1; let b = 2; let c = 1; let d = 2; [&mut a, &mut b] == [&mut c, &mut d]";
    let reads = mut_parts_read_shared(source, |op| matches!(op, Op::Index { .. }));
    assert_eq!(reads, 2, "one element read on each side, in the loop");
}

/// ownership §5, types §4, D108: a `&mut` part of a tuple compared is read
/// through the tuple's shared reference as a shared reborrow.
#[test]
fn a_tuple_with_a_mut_part_compares_it_by_shared_reborrow() {
    let source = "let a = 1; let b = 1; (&mut a, true) == (&mut b, true)";
    let reads = mut_parts_read_shared(source, |op| matches!(op, Op::Take { target: RefTarget::Through(_), path, .. } if !path.is_empty()));
    assert_eq!(reads, 2, "one part read on each side");
}

/// `source` lowered, each body held to `TypesAgree` among the rest, with
/// `$s` a reference of `slice` to a slice of `&mut i64` where given; the
/// kinds of the reference values the entry's reads `read` picks yield.
fn part_reads(source: &str, slice: Option<Mutability>, read: impl Fn(&Op) -> bool) -> Vec<TyKind<Concrete>> {
    let mut fx = Fixture::new();
    if let Some(held) = slice {
        let int = fx.int();
        let element = fx.reference(Mutability::Mut, int);
        let slice_ty = fx.ty(TyKind::Slice(element));
        let lent = fx.reference(held, slice_ty);
        fx.input("s", lent);
    }
    let lowered = fx.script(source);
    let body = lowered.entry();
    read_values(body, &read)
        .into_iter()
        .map(|value| lowered.kind_of(body, value).clone())
        .filter(|kind| matches!(kind, TyKind::Ref(..)))
        .collect()
}

/// The value each read `read` picks yields: the reference it takes out of
/// its place, or, where that is taken as a weaker reborrow past its path,
/// the reborrow made through it (ownership §5).
fn read_values(body: &Body, read: &impl Fn(&Op) -> bool) -> Vec<crate::ir::ValueId> {
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let reborrow_of = |taken: crate::ir::ValueId| {
        insts.iter().find_map(|inst| match inst.op() {
            Op::Ref { target: RefTarget::Through(through), path, .. } if *through == taken && path.is_empty() => inst.result(),
            _ => None,
        })
    };
    insts
        .iter()
        .filter(|inst| read(inst.op()))
        .filter_map(|inst| inst.result())
        .map(|taken| reborrow_of(taken).unwrap_or(taken))
        .collect()
}

fn is_reference(kind: &TyKind<Concrete>, mutability: Mutability) -> bool {
    matches!(kind, TyKind::Ref(made, _) if *made == mutability)
}

/// D108 rule 1: a `&mut` element read through a shared slice is copied
/// out as a shared reborrow.
#[test]
fn an_index_of_a_shared_slice_of_mut_references_lowers_to_a_shared_read() {
    let reads = part_reads("let x = $s[0]; *x", Some(Mutability::Shared), |op| matches!(op, Op::Index { .. }));
    assert!(matches!(&reads[..], [read] if is_reference(read, Mutability::Shared)), "{reads:?}");
}

/// D108 rule 1: through a `&mut` slice the element is read as a `&mut`.
#[test]
fn an_index_of_a_mut_slice_of_mut_references_lowers_to_a_mut_read() {
    let reads = part_reads("let x = $s[0]; *x", Some(Mutability::Mut), |op| matches!(op, Op::Index { .. }));
    assert!(matches!(&reads[..], [read] if is_reference(read, Mutability::Mut)), "{reads:?}");
}

/// D108 rule 1: a `&mut` field past a `&` is read as a shared reborrow.
#[test]
fn a_mut_field_past_a_shared_reference_lowers_to_a_shared_read() {
    let source = "let v = 1; let o = { a: { b: &mut v, }, }; let r = &o; let x = r.a.b; *x";
    let reads = part_reads(source, None, |op| matches!(op, Op::Take { target: RefTarget::Through(_), path, .. } if path.len() == 2));
    assert!(matches!(&reads[..], [read] if is_reference(read, Mutability::Shared)), "{reads:?}");
}

/// D108 rule 1: a `&mut` reached past a `&` and dereferenced on the way is
/// read shared, and so is the `&mut` field past it.
#[test]
fn a_mut_reference_on_the_way_past_a_shared_one_lowers_to_shared_reads() {
    let source = "let v = 1; let w = { b: &mut v, }; let o = { a: &mut w, }; let r = &o; let x = r.a.b; *x";
    let reads = part_reads(source, None, |op| matches!(op, Op::Take { target: RefTarget::Through(_), path, .. } if !path.is_empty()));
    assert!(matches!(&reads[..], [on_the_way, read] if [on_the_way, read].iter().all(|kind| is_reference(kind, Mutability::Shared))), "{reads:?}");
}

/// The kinds of the references a pattern binding in `source` reads out
/// of a place under a reference, each body held to `TypesAgree`.
fn bound_part_reads(source: &str) -> Vec<TyKind<Concrete>> {
    part_reads(source, None, |op| matches!(op, Op::Take { target: RefTarget::Through(_), path, .. } if !path.is_empty()))
}

/// D108 rule 1, patterns §1: a `&mut` part bound past a `&` is read as a
/// shared reborrow.
#[test]
fn a_mut_part_bound_past_a_shared_reference_lowers_to_a_shared_read() {
    let reads = bound_part_reads("let v = 1; let o = { b: &mut v, }; match &o { { b, } => *b, }");
    assert!(matches!(&reads[..], [read] if is_reference(read, Mutability::Shared)), "{reads:?}");
}

/// D108 rule 1, patterns §1: past a `&mut` the part is read as a `&mut`.
#[test]
fn a_mut_part_bound_past_a_mut_reference_lowers_to_a_mut_read() {
    let reads = bound_part_reads("let v = 1; let o = { b: &mut v, }; match &mut o { { b, } => *b, }");
    assert!(matches!(&reads[..], [read] if is_reference(read, Mutability::Mut)), "{reads:?}");
}

/// D108 rule 1, patterns §1: a `&` part bound past a `&mut` is read as
/// the `&` it is.
#[test]
fn a_shared_part_bound_past_a_mut_reference_lowers_to_a_shared_read() {
    let reads = bound_part_reads("let v = 1; let o = { b: &v, }; match &mut o { { b, } => *b, }");
    assert!(matches!(&reads[..], [read] if is_reference(read, Mutability::Shared)), "{reads:?}");
}

/// D108 rule 1, patterns §1: a `&mut` a match passes on the way past a `&`
/// is read shared, and so is the `&mut` part bound past it.
#[test]
fn a_mut_reference_a_match_passes_past_a_shared_one_lowers_to_shared_reads() {
    let source = "let v = 1; let w = { b: &mut v, }; let o = { a: &mut w, }; match &o { { a: { b, }, } => *b, }";
    let reads = bound_part_reads(source);
    assert!(matches!(&reads[..], [on_the_way, read] if [on_the_way, read].iter().all(|kind| is_reference(kind, Mutability::Shared))), "{reads:?}");
}

/// D108 rule 1: a store into a `&mut` field through a `&mut` writes the
/// field at its own type.
#[test]
fn a_store_into_a_mut_field_through_a_mut_reference_lowers() {
    let source = "let v = 1; let w = 2; let o = { f: &mut v, }; let r = &mut o; r.f = &mut w; *o.f";
    let fx = Fixture::new();
    let lowered = fx.script(source);
    let stores = count(&lowered.entry_ops(), |op| matches!(op, Op::Assign { target: RefTarget::Through(_), path, .. } if path == &[PathSeg::Field(lowered.name("f"))]));
    assert_eq!(stores, 1, "{}", shown(lowered.entry()));
}

/// operators §2, §4, D87: an ordering on arrays is one `For` that leaves
/// at the first `cmp` that is not `0`, and the answer is compared with `0`
/// after it.
#[test]
fn array_ordering_is_one_for_that_exits_at_the_first_nonzero_cmp() {
    let mut fx = Fixture::new();
    fx.comparisons();
    let int = fx.int();
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(4), ()));
    fx.input("a", array);
    fx.input("b", array);
    let lowered = fx.script("$a < $b");
    let body = lowered.entry();
    let [(_, _, exit)] = fors(&lowered)[..] else { panic!("one For: {}", shown(body)) };
    let [answer] = body.block(exit).params() else { panic!("the exit takes the answer: {}", shown(body)) };
    assert_eq!(lowered.kind_of(body, *answer), &TyKind::Int(IntTy::I64));
    let cmp = QualifiedRef::qualified(lowered.namespace(&["core"]), lowered.name("cmp"));
    assert_eq!(calls_of(&lowered, cmp), 1, "{}", shown(body));
    let lt = count(&lowered.entry_ops(), |op| matches!(op, Op::BinOp { op: crate::ir::BinOp::Lt, left, .. } if left == answer));
    assert_eq!(lt, 1, "{}", shown(body));
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
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(3), ()));
    let pair = fx.ty(TyKind::Tuple(Box::new([string, int]), ()));
    fx.input("a", array);
    fx.input("b", array);
    fx.input("t", pair);
    let script: Script = acvus_ast::parse_script(&fx.names, "let c = $t.clone(); $a < $b").expect("parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let mut types = Types::new();
    let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &fx.inputs);
    let typeck = check_script(&fx.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &script);
    assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
    let mut module = lower_script(&decls, &typeck, types, &script);
    let mut am = crate::passes::analysis::AnalysisManager::new(&decls);
    super::required::run(&mut module, &mut am).expect("the checks admit it");
    crate::passes::pipeline::optimize(&mut module, &mut am, crate::passes::pipeline::Level::Full);
    Lowered::checked(module, fx.names);
}

/// `source` through typeck, lower, the required steps with the checks, and
/// the full pipeline: the checks' refusals, and none past them.
fn through_the_pipeline(fx: Fixture, source: &str) -> Result<(), Vec<crate::check::Refusal>> {
    let script: Script = acvus_ast::parse_script(&fx.names, source).expect("parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let mut types = Types::new();
    let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &fx.inputs);
    let typeck = check_script(&fx.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &script);
    assert!(typeck.refusals.is_empty(), "refused: {:?}", typeck.refusals);
    let mut module = lower_script(&decls, &typeck, types, &script);
    let mut am = crate::passes::analysis::AnalysisManager::new(&decls);
    super::required::run(&mut module, &mut am)?;
    crate::passes::pipeline::optimize(&mut module, &mut am, crate::passes::pipeline::Level::Full);
    Lowered::checked(module, fx.names);
    Ok(())
}

/// A program the checks admit goes on through the drops and the pipeline;
/// one they refuse stops at them with its refusals (refusals).
#[test]
fn the_required_steps_run_the_checks_before_the_drops() {
    let admitted = "let o = { a: 1, b: 2, }; let r = &mut o.a; *r = 3; o.b = 4; o";
    assert_eq!(through_the_pipeline(Fixture::new(), admitted), Ok(()));
    let refused = "let a; let b = a; a = 1; b";
    let Err(refusals) = through_the_pipeline(Fixture::new(), refused) else {
        panic!("a read before any store is refused");
    };
    assert!(matches!(refusals[..], [crate::check::Refusal { kind: crate::check::RefusalKind::NotWritten, .. }]), "{refusals:?}");
}

// Structured coercion and lifted impls (types §1, §3)

/// types §3: an object entering a declaration's parameter is narrowed
/// there: a new object of the parameter's fields, each taken from the
/// value, is what the call receives.
#[test]
fn a_narrowed_object_is_a_new_object_of_the_parameters_fields() {
    let mut fx = Fixture::new();
    let a = fx.name("a");
    let (int, unit) = (fx.int(), fx.ty(TyKind::Unit));
    let object = fx.ty(TyKind::Object(crate::ty::ObjectTy::Structural(Box::new([(a, int)]), ())));
    let ty = fx.function(&[object], unit, Effect::pure());
    let f = fx.declare("f", ty, 1);
    let lowered = fx.script("f({ a: 1, b: true, })");
    let body = lowered.entry();
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::FieldGet { field, .. } if *field == a)), 1, "{}", shown(body));
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let call = insts
        .iter()
        .find(|inst| matches!(inst.op(), Op::FunctionCall { callee: Callee::Extern { id, .. }, .. } if *id == f))
        .expect("the call of `f`");
    let Op::FunctionCall { args, .. } = call.op() else {
        unreachable!("found as a call");
    };
    let TyKind::Object(narrowed) = lowered.kind_of(body, args[0]) else {
        panic!("`f` receives an object: {}", shown(body));
    };
    assert_eq!(narrowed.fields().iter().map(|(name, _)| *name).collect::<Vec<_>>(), vec![a], "{}", shown(body));
    let made = insts.iter().find(|inst| inst.result() == Some(args[0])).expect("the argument is made in the body");
    assert!(matches!(made.op(), Op::MakeObject { fields } if fields.len() == 1), "{}", shown(body));
}

/// types §1, ownership §2: a tuple's lifted `clone` calls its `String`
/// part's own impl on a reference to that part, copies its `i64` part,
/// which copies, and makes a tuple of the two.
#[test]
fn a_lifted_clone_calls_each_parts_impl_and_rebuilds_the_tuple() {
    let mut fx = Fixture::new();
    let (int, string) = (fx.int(), fx.ty(TyKind::String));
    let impls: Vec<TyId<Poly>> = [string, int]
        .into_iter()
        .map(|ty| {
            let shared = fx.reference(Mutability::Shared, ty);
            fx.function(&[shared], ty, Effect::pure())
        })
        .collect();
    let clone = fx.raw.core.clone;
    fx.lang_item(clone, &impls);
    let pair = fx.ty(TyKind::Tuple(Box::new([string, int]), ()));
    fx.input("t", pair);
    let lowered = fx.script("$t.clone()");
    let body = lowered.entry();
    let ops = lowered.entry_ops();
    let calls: Vec<Option<usize>> = ops
        .iter()
        .filter_map(|op| match op {
            Op::FunctionCall { callee: Callee::Extern { id, r#impl, .. }, .. } if *id == clone => Some(*r#impl),
            _ => None,
        })
        .collect();
    assert_eq!(calls, vec![Some(0)], "only the `String` part calls an impl: {}", shown(body));
    for at in [0, 1] {
        let part = |op: &Op| matches!(op, Op::Ref { path, mutability: Mutability::Shared, .. } if path[..] == [PathSeg::Index(at)]);
        assert_eq!(count(&ops, part), 1, "part {at} is lent: {}", shown(body));
    }
    let copied = |op: &Op| matches!(op, Op::Take { target: RefTarget::Through(_), path, .. } if path.is_empty());
    assert_eq!(count(&ops, copied), 1, "the `i64` part is its copy: {}", shown(body));
    assert_eq!(count(&ops, |op| matches!(op, Op::MakeTuple { elements } if elements.len() == 2)), 1, "{}", shown(body));
}

/// types §1: a tuple's lifted `eq` asks each part's own impl, the second
/// only where the first answered equal.
#[test]
fn a_lifted_eq_asks_each_parts_impl_in_order() {
    let mut fx = Fixture::new();
    let (int, bool_ty) = (fx.int(), fx.ty(TyKind::Bool));
    let shared = fx.reference(Mutability::Shared, int);
    let at_int = fx.function(&[shared, shared], bool_ty, Effect::pure());
    let eq = fx.raw.core.eq;
    let facts = DeclaredFacts { access: Box::new([crate::decl::declared::Access::Equivalence]), ..Default::default() };
    let concrete = Box::new([ImplSig { ty: at_int, bounds: Box::new([]), param_names: fx.param_names(2), requires: Box::new([]), facts, runs: Box::new([]) }]);
    fx.raw.lang_impls.push((eq, Impls { concrete, generic: None }));
    let pair = fx.ty(TyKind::Tuple(Box::new([int, int]), ()));
    fx.input("t", pair);
    let lowered = fx.script("eq(&$t, &$t)");
    let body = lowered.entry();
    let ops = lowered.entry_ops();
    let calls = count(&ops, |op| matches!(op, Op::FunctionCall { callee: Callee::Extern { id, .. }, .. } if *id == eq));
    assert_eq!(calls, 2, "{}", shown(body));
    let diamonds = lowered.entry_terminators().iter().filter(|terminator| matches!(terminator, Terminator::Diamond { .. })).count();
    assert_eq!(diamonds, 1, "the second part is asked only where the first is equal: {}", shown(body));
}

/// The instructions of every body of `lowered`, the entry's first.
fn all_ops(lowered: &Lowered) -> Vec<&Op> {
    lowered.module.body_ids().flat_map(|id| lowered.ops(id)).collect()
}

/// types §3, ownership §5: a reference to an object entering a parameter
/// that is a reference to a narrower one is a reference into the original
/// storage for each field the parameter names, gathered into the
/// reference the call receives; nothing is taken out or rebuilt.
#[test]
fn a_reference_coerced_through_its_target_gathers_references_to_its_parts() {
    let mut fx = Fixture::new();
    let a = fx.name("a");
    let (int, unit) = (fx.int(), fx.ty(TyKind::Unit));
    let object = fx.ty(TyKind::Object(crate::ty::ObjectTy::Structural(Box::new([(a, int)]), ())));
    let shared = fx.reference(Mutability::Shared, object);
    let ty = fx.function(&[shared], unit, Effect::pure());
    fx.declare("f", ty, 1);
    let lowered = fx.script("f(&{ a: 1, b: true, })");
    let body = lowered.entry();
    let ops = lowered.entry_ops();
    let lent = |op: &Op| matches!(op, Op::Ref { target: RefTarget::Through(_), path, mutability: Mutability::Shared } if path[..] == [PathSeg::Field(a)]);
    assert_eq!(count(&ops, lent), 1, "{}", shown(body));
    let gathered = |op: &Op| matches!(op, Op::Intrinsic { intrinsic: crate::ir::Intrinsic::Gather, .. });
    assert_eq!(count(&ops, gathered), 1, "{}", shown(body));
    assert_eq!(count(&ops, |op| matches!(op, Op::FieldGet { .. })), 0, "nothing is taken out: {}", shown(body));
}

/// types §1, §3: an array stored into a context of another element type
/// is mapped element by element, never unrolled.
#[test]
fn an_array_coerced_is_mapped_element_by_element() {
    let mut fx = Fixture::new();
    let (u8_ty, int) = (fx.ty(TyKind::Int(IntTy::U8)), fx.int());
    let widening = fx.function(&[u8_ty], int, Effect::pure());
    let widen = fx.declare("widen", widening, 1);
    fx.raw.casts.push(crate::decl::CastRule { from: u8_ty, to: int, via: widen });
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(2), ()));
    fx.context("v", array, false);
    let lowered = fx.script("@v = [1u8, 2u8]; 0");
    let body = lowered.entry();
    let ops = lowered.entry_ops();
    let mapped = |op: &Op| matches!(op, Op::Intrinsic { intrinsic: crate::ir::Intrinsic::ArrayMap, .. });
    assert_eq!(count(&ops, mapped), 1, "{}", shown(body));
    let widened = |op: &Op| matches!(op, Op::FunctionCall { callee: Callee::Extern { id, .. }, .. } if *id == widen);
    assert_eq!(count(&all_ops(&lowered), widened), 1, "the cast is made once, in the map's function: {}", shown(body));
}

/// types §1: an array's lifted `clone` maps its elements by the element's
/// impl.
#[test]
fn a_lifted_clone_of_an_array_maps_its_elements() {
    let mut fx = Fixture::new();
    let string = fx.ty(TyKind::String);
    let shared = fx.reference(Mutability::Shared, string);
    let clones = fx.function(&[shared], string, Effect::pure());
    let clone = fx.raw.core.clone;
    fx.lang_item(clone, &[clones]);
    let array = fx.ty(TyKind::Array(string, crate::ty::LenTerm::Known(2), ()));
    fx.input("a", array);
    let lowered = fx.script("$a.clone()");
    let body = lowered.entry();
    let mapped = |op: &Op| matches!(op, Op::Intrinsic { intrinsic: crate::ir::Intrinsic::ArrayMap, .. });
    assert_eq!(count(&lowered.entry_ops(), mapped), 1, "{}", shown(body));
    let cloned = |op: &Op| matches!(op, Op::FunctionCall { callee: Callee::Extern { id, .. }, .. } if *id == clone);
    assert_eq!(count(&all_ops(&lowered), cloned), 1, "{}", shown(body));
}

/// ownership §2, operators §2: a `clone` of a value that copies is its
/// copy, and `Option`'s `default` is `None`; neither calls an impl.
#[test]
fn the_languages_own_answers_call_no_impl() {
    let lowered = Fixture::new().script("let x = 1.clone(); x");
    let body = lowered.entry();
    let ops = lowered.entry_ops();
    assert_eq!(count(&ops, |op| matches!(op, Op::FunctionCall { .. })), 0, "{}", shown(body));
    let copied = |op: &Op| matches!(op, Op::Take { target: RefTarget::Through(_), path, .. } if path.is_empty());
    assert_eq!(count(&ops, copied), 1, "{}", shown(body));

    let mut fx = Fixture::new();
    let (int, unit) = (fx.int(), fx.ty(TyKind::Unit));
    let option = fx.ty(TyKind::Option(int, ()));
    let ty = fx.function(&[option], unit, Effect::pure());
    fx.declare("f", ty, 1);
    let lowered = fx.script("f(default())");
    let body = lowered.entry();
    let none = lowered.name("None");
    let made = |op: &Op| matches!(op, Op::MakeVariant { tag, payload: None } if *tag == none);
    assert_eq!(count(&lowered.entry_ops(), made), 1, "{}", shown(body));
}

/// types §1: an array rebuilt from no argument, as `default` makes one, is
/// `ArrayMap` of no array, its length the array type's; from two, one map
/// of both.
#[test]
fn an_array_rebuilt_from_no_argument_or_several_is_one_map() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let at_int = fx.function(&[], int, Effect::pure());
    let default = fx.raw.core.default;
    fx.lang_item(default, &[at_int]);
    let array = fx.ty(TyKind::Array(int, crate::ty::LenTerm::Known(3), ()));
    fx.context("v", array, false);
    let lowered = fx.script("@v = default(); 0");
    let body = lowered.entry();
    let mapped = |op: &Op| matches!(op, Op::Intrinsic { intrinsic: crate::ir::Intrinsic::ArrayMap, args } if args.len() == 1);
    assert_eq!(count(&lowered.entry_ops(), mapped), 1, "{}", shown(body));

    let mut fx = Fixture::new();
    let int = fx.int();
    let t = fx.ty(TyKind::Var(0));
    let (shared_t, shared_int) = (fx.reference(Mutability::Shared, t), fx.reference(Mutability::Shared, int));
    let generic = fx.function(&[shared_t, shared_t], t, Effect::pure());
    let at_int = fx.function(&[shared_int, shared_int], int, Effect::pure());
    let id = QualifiedRef::root(fx.name("pick"));
    let concrete = Box::new([ImplSig { ty: at_int, bounds: Box::new([]), param_names: fx.param_names(2), requires: Box::new([]), facts: DeclaredFacts::default(), runs: Box::new([]) }]);
    let scheme = Scheme {
        ty: generic,
        param_names: fx.param_names(2),
        bounds: Box::new([TyVarBound::Any]),
        impls: Some(Impls { concrete, generic: None }),
        requires: Box::new([]),
        facts: DeclaredFacts::default(),
        runs: Box::new([]),
    };
    fx.raw.functions.push((id, scheme));
    let lowered = fx.script("let a = [1, 2]; let b = [3, 4]; let c = pick(&a, &b); 0");
    let body = lowered.entry();
    let mapped = |op: &Op| matches!(op, Op::Intrinsic { intrinsic: crate::ir::Intrinsic::ArrayMap, args } if args.len() == 3);
    assert_eq!(count(&lowered.entry_ops(), mapped), 1, "{}", shown(body));
}

/// operators §2, ownership §5: `as_ref` lowers to a reference into each
/// part of the original storage, rebuilt into the structure: nothing is
/// taken out of it or copied, and an array is one map, not unrolled.
#[test]
fn as_ref_lowers_to_references_into_each_part() {
    let lowered = Fixture::new().script("let p = (1, true); let r = p.as_ref(); 0");
    let body = lowered.entry();
    let ops = lowered.entry_ops();
    let into_part = |op: &Op| matches!(op, Op::Ref { target: RefTarget::Through(_), path, mutability: Mutability::Shared } if path.len() == 1);
    assert_eq!(count(&ops, into_part), 2, "{}", shown(body));
    let taken_whole = |op: &Op| matches!(op, Op::Take { target: RefTarget::Through(_), path, .. } if path.is_empty());
    assert_eq!(count(&ops, taken_whole), 0, "{}", shown(body));
    assert_eq!(count(&ops, |op| matches!(op, Op::MakeTuple { elements } if elements.len() == 2)), 2, "{}", shown(body));

    let lowered = Fixture::new().script("let o = Some(1); let r = o.as_ref(); 0");
    let ops = lowered.entry_ops();
    let payload = |op: &Op| matches!(op, Op::Ref { path, .. } if path[..] == [PathSeg::Payload]);
    assert_eq!(count(&ops, payload), 1, "{}", shown(lowered.entry()));

    let lowered = Fixture::new().script("let a = [1, 2]; let r = a.as_ref(); 0");
    let mapped = |op: &Op| matches!(op, Op::Intrinsic { intrinsic: crate::ir::Intrinsic::ArrayMap, args } if args.len() == 2);
    assert_eq!(count(&lowered.entry_ops(), mapped), 1, "{}", shown(lowered.entry()));
}

/// types §4, ownership §5: over a reference to an array of references,
/// `ArrayMap` passes each element reborrowed at the weaker mutability, as
/// the IR type check holds it to.
#[test]
fn a_lend_of_an_array_of_references_maps_each_reborrowed() {
    for source in [
        "let x = 1; let y = 2; let a = [&mut x, &mut y]; let r = a.as_ref(); 0",
        "let x = 1; let y = 2; let a = [&x, &y]; let r = a.as_mut(); 0",
    ] {
        let lowered = Fixture::new().script(source);
        let mapped = |op: &Op| matches!(op, Op::Intrinsic { intrinsic: crate::ir::Intrinsic::ArrayMap, .. });
        assert_eq!(count(&lowered.entry_ops(), mapped), 1, "{source}: {}", shown(lowered.entry()));
    }
}

// G, a generalized binding's code per signature (functions §2)

/// Each `MakeClosure` of the entry body, with its bodies and captures.
fn made_closures(lowered: &Lowered) -> Vec<(Vec<BodyId>, usize)> {
    lowered
        .entry_ops()
        .into_iter()
        .filter_map(|op| match op {
            Op::MakeClosure { bodies, captures } => Some((bodies.clone(), captures.len())),
            _ => None,
        })
        .collect()
}

/// functions §2: a generalized binding used at two signatures is one
/// closure over its environment, taken once, with a body at each
/// signature, and each use keeps it at its own.
#[test]
fn a_generalized_lambda_has_a_body_per_signature_over_one_environment() {
    let lowered = Fixture::new().script("let k = 2; let f = |x| -> (x, k); let a = f(1); let b = f(true); 0");
    let body = lowered.entry();
    let closures = made_closures(&lowered);
    let [(bodies, captures)] = &closures[..] else {
        panic!("one closure: {}", shown(body));
    };
    assert_eq!(bodies.len(), 2, "{}", shown(body));
    assert_eq!(*captures, 1, "the environment is taken once: {}", shown(body));
    let specialized = count(&lowered.entry_ops(), |op| matches!(op, Op::Specialize { signatures, .. } if signatures.len() == 1));
    assert_eq!(specialized, 2, "{}", shown(body));
}

/// functions §2: two uses at one signature share one body, and a closure
/// of one signature is that function type, used as it is.
#[test]
fn uses_at_one_signature_share_one_body() {
    let lowered = Fixture::new().script("let f = |x| -> x; let a = f(1); let b = f(2); 0");
    let body = lowered.entry();
    let closures = made_closures(&lowered);
    let [(bodies, _)] = &closures[..] else {
        panic!("one closure: {}", shown(body));
    };
    assert_eq!(bodies.len(), 1, "{}", shown(body));
    assert_eq!(count(&lowered.entry_ops(), |op| matches!(op, Op::Specialize { .. })), 0, "{}", shown(body));
}

/// functions §2: the lambdas joined into one generalized binding have one
/// type, the closure type of its set of signatures, so either stands where
/// the other would.
#[test]
fn a_join_of_lambdas_is_of_one_closure_type() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let lowered = fx.script("let x = if $c { |v| -> (v, v) } else { |v| -> (v, v) }; let p = x(10); let q = x(true); 0");
    let body = lowered.entry();
    let made: Vec<crate::ir::ValueId> = body
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .filter(|inst| matches!(inst.op(), Op::MakeClosure { .. }))
        .filter_map(|inst| inst.result())
        .collect();
    let [left, right] = made[..] else {
        panic!("two closures: {}", shown(body));
    };
    assert_eq!(body.ty_of(left), body.ty_of(right), "one type: {}", shown(body));
    let TyKind::Closures(signatures) = lowered.kind_of(body, left) else {
        panic!("a closure of signatures: {}", shown(body));
    };
    assert_eq!(signatures.len(), 2, "{}", shown(body));
}

/// functions §2: a generalized binding passed where one function type is
/// taken is kept at that one signature.
#[test]
fn a_generalized_binding_passed_as_a_value_is_specialized_to_its_parameter() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let taken = fx.function(&[int], int, Effect::pure());
    let ty = fx.function(&[taken], int, Effect::pure());
    fx.declare("apply", ty, 1);
    let lowered = fx.script("let f = |x| -> x; let a = apply(f); let b = f(true); 0");
    let body = lowered.entry();
    let specialized: Vec<&Op> = lowered.entry_ops().into_iter().filter(|op| matches!(op, Op::Specialize { .. })).collect();
    assert_eq!(specialized.len(), 2, "{}", shown(body));
    let apply_arg = body
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .find_map(|inst| match inst.op() {
            Op::FunctionCall { callee: Callee::Extern { .. }, args, .. } => Some(args[0]),
            _ => None,
        })
        .expect("the call of `apply`");
    assert!(matches!(lowered.kind_of(body, apply_arg), TyKind::Fn { params, .. } if params.len() == 1), "{}", shown(body));
}

/// functions §2, D67 item 9: `&f` of a generalized binding lends the value
/// its name reads, kept at the use's signature and put in a slot of its
/// own; the binding's storage is never lent.
#[test]
fn a_borrow_of_a_generalized_binding_lends_a_specialized_value() {
    let mut fx = Fixture::new();
    let (int, bool_ty) = (fx.int(), fx.ty(TyKind::Bool));
    for (name, ty) in [("on_int", int), ("on_bool", bool_ty)] {
        let taken = fx.function(&[ty], ty, Effect::pure());
        let lent = fx.reference(Mutability::Shared, taken);
        let applies = fx.function(&[lent], ty, Effect::pure());
        fx.declare(name, applies, 1);
    }
    let lowered = fx.script("let f = |x| -> x; let a = on_int(&f); let b = on_bool(&f); 0");
    let body = lowered.entry();
    let insts: Vec<_> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
    let specialized: Vec<crate::ir::ValueId> =
        insts.iter().filter(|inst| matches!(inst.op(), Op::Specialize { .. })).filter_map(|inst| inst.result()).collect();
    assert_eq!(specialized.len(), 2, "{}", shown(body));
    for value in specialized {
        let stored = insts.iter().any(|inst| matches!(inst.op(), Op::Assign { target: RefTarget::Var(_), path, value: stored, .. } if path.is_empty() && *stored == value));
        assert!(stored, "the specialized value is put in a slot and lent from there: {}", shown(body));
    }
}

/// functions §2: a generalized binding's value passing through a `match`
/// and a block is of its closure type at every site typeck recorded it
/// passes through, with no form inspected by lower.
#[test]
fn a_generalized_value_through_a_match_and_a_block_is_of_its_closure_type() {
    let source = "let x = match 1 { 1 => { let z = 0; |v| -> (v, v) }, _ => |v| -> (v, v), }; let p = x(10); let q = x(true); 0";
    let lowered = Fixture::new().script(source);
    let body = lowered.entry();
    let made: Vec<crate::ir::ValueId> = body
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .filter(|inst| matches!(inst.op(), Op::MakeClosure { .. }))
        .filter_map(|inst| inst.result())
        .collect();
    let [left, right] = made[..] else {
        panic!("two closures: {}", shown(body));
    };
    assert_eq!(body.ty_of(left), body.ty_of(right), "{}", shown(body));
    assert!(matches!(lowered.kind_of(body, left), TyKind::Closures(signatures) if signatures.len() == 2), "{}", shown(body));
}

/// functions §2: a generalized binding no use reaches is still made where
/// it is written, since making a closure takes its captures: a closure of
/// no signature, with no code and only its environment.
#[test]
fn an_unused_generalized_binding_makes_a_closure_of_no_signature() {
    let lowered = Fixture::new().script("let n = 1; let f = |x| -> n; 0");
    let body = lowered.entry();
    let made: Vec<_> = body
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .filter_map(|inst| match inst.op() {
            Op::MakeClosure { bodies, captures } => Some((inst.result(), bodies.len(), captures.len())),
            _ => None,
        })
        .collect();
    let [(Some(closure), 0, captures)] = made[..] else {
        panic!("one closure of no body: {}", shown(body));
    };
    assert_eq!(captures, 1, "its capture is taken: {}", shown(body));
    assert!(matches!(lowered.kind_of(body, closure), TyKind::Closures(signatures) if signatures.is_empty()), "{}", shown(body));
}
