//! `#τ` is a type (scratchpad/tobe/hash-types.md), at the contract: the
//! representation an extern signature names is what the checker infers,
//! two representations at one specializing position are a conversion or
//! an error, the demand reaches the producer, and the generic instance is
//! the uniform one.

use acvus_extern::{Externs, Monomorphize, Registry, TypesOnly, extern_fn, extern_registry};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ir::Callee;
use acvus_mir::ty::{
    CastRule, Effect, Instances, LenTerm, ParamTerm, Poly, PolyBuilder, PolyTy, Repr, Ty, TyTerm,
    TyVarBound, TypeArg, TypeRegistry, UserDefinedDecl,
};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

// -- Types ------------------------------------------------------------

fn vec_ref(i: &Interner) -> QualifiedRef {
    QualifiedRef::root(i.intern("Vec"))
}

fn box2_ref(i: &Interner) -> QualifiedRef {
    QualifiedRef::root(i.intern("Box2"))
}

fn vec_of<V>(i: &Interner, arg: TypeArg<V>) -> TyTerm<V>
where
    V: acvus_mir::ty::Phase,
{
    TyTerm::UserDefined {
        id: vec_ref(i),
        type_args: vec![arg],
        effect_args: vec![],
        identity_args: vec![],
    }
}

fn box2_of<V>(i: &Interner, arg: TypeArg<V>) -> TyTerm<V>
where
    V: acvus_mir::ty::Phase,
{
    TyTerm::UserDefined {
        id: box2_ref(i),
        type_args: vec![arg],
        effect_args: vec![],
        identity_args: vec![],
    }
}

/// A type variable at a specializing position (hash-types.md, Signatures).
fn var_arg(t: &PolyTy) -> TypeArg<Poly> {
    let TyTerm::Var(v) = t else {
        panic!("a type variable")
    };
    TypeArg::new(Repr::Var(*v), t.clone())
}

fn spec(ty: PolyTy) -> TypeArg<Poly> {
    TypeArg::specialized(ty)
}

fn unif(ty: PolyTy) -> TypeArg<Poly> {
    TypeArg::uniform(ty)
}

fn registry(i: &Interner) -> TypeRegistry {
    let mut reg = TypeRegistry::new();
    reg.register(UserDefinedDecl {
        qref: vec_ref(i),
        type_params: vec![TyVarBound::Any],
        effect_params: 0,
        identity_params: 0,
        specializable: vec![true],
    });
    reg.register(UserDefinedDecl {
        qref: box2_ref(i),
        type_params: vec![TyVarBound::Any],
        effect_params: 0,
        identity_params: 0,
        specializable: vec![false],
    });
    reg
}

fn with_casts(i: &Interner, mut reg: TypeRegistry) -> TypeRegistry {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    reg.register_cast(CastRule {
        from: vec_of(i, spec(t.clone())),
        to: vec_of(i, unif(t.clone())),
        fn_ref: QualifiedRef::root(i.intern("vec_erase")),
    });
    reg.register_cast(CastRule {
        from: vec_of(i, unif(t.clone())),
        to: vec_of(i, spec(t)),
        fn_ref: QualifiedRef::root(i.intern("vec_materialize")),
    });
    reg
}

// -- Externs ----------------------------------------------------------

fn fn_of(i: &Interner, params: &[(&str, PolyTy)], ret: PolyTy) -> PolyTy {
    TyTerm::Fn {
        params: params
            .iter()
            .map(|(n, t)| ParamTerm::<Poly>::new(i.intern(n), t.clone()))
            .collect(),
        ret: Box::new(ret),
        captures: vec![],
        effect: Effect::PURE.into(),
    }
}

fn extern_fn(i: &Interner, name: &str, ty: PolyTy, instances: Instances) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances,
        },
        ty,
    }
}

fn concrete(i: &Interner, name: &str, ty: PolyTy) -> Function {
    extern_fn(i, name, ty, Instances::default())
}

/// The externs of the evaluation, as their instances hold them.
fn externs(i: &Interner) -> Vec<Function> {
    let string = || TyTerm::String;
    let int = || TyTerm::I64;
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    let n = pb.fresh_len_var();
    let array_of = |elem: PolyTy| TyTerm::Array(Box::new(elem), n.clone());
    let option_of = |elem: PolyTy| TyTerm::Option(Box::new(elem));

    let vec_array_generic = fn_of(i, &[("items", array_of(t.clone()))], vec_of(i, var_arg(&t)));
    let vec_array_at =
        |elem: TypeArg<Poly>| fn_of(i, &[("items", array_of(string()))], vec_of(i, elem));
    let mut vec_array_concrete = vec![TyTerm::Unit; VEC_ARRAY_CONCRETE];
    vec_array_concrete[VEC_ARRAY_AT_SPECIALIZED] = vec_array_at(spec(string()));
    vec_array_concrete[VEC_ARRAY_AT_UNIFORM] = vec_array_at(unif(string()));
    let mut reverse_concrete = vec![TyTerm::Unit; REVERSE_CONCRETE];
    reverse_concrete[REVERSE_AT_SPECIALIZED] = fn_of(
        i,
        &[("items", vec_of(i, spec(string())))],
        vec_of(i, spec(string())),
    );
    let reverse_generic = fn_of(
        i,
        &[("items", vec_of(i, var_arg(&t)))],
        vec_of(i, var_arg(&t)),
    );
    let first_generic = fn_of(i, &[("items", vec_of(i, var_arg(&t)))], t.clone());

    vec![
        concrete(i, "f", fn_of(i, &[], vec_of(i, spec(string())))),
        concrete(i, "k", fn_of(i, &[], vec_of(i, unif(string())))),
        concrete(i, "g", fn_of(i, &[("v", vec_of(i, spec(string())))], int())),
        concrete(i, "m", fn_of(i, &[("v", vec_of(i, unif(string())))], int())),
        extern_fn(
            i,
            "vec_array",
            vec_array_generic,
            Instances {
                concrete: vec_array_concrete,
                generic: true,
            },
        ),
        extern_fn(
            i,
            "reverse",
            reverse_generic,
            Instances {
                concrete: reverse_concrete,
                generic: true,
            },
        ),
        extern_fn(
            i,
            "first",
            first_generic,
            Instances {
                concrete: vec![],
                generic: true,
            },
        ),
        concrete(
            i,
            "opt",
            fn_of(i, &[], vec_of(i, spec(option_of(string())))),
        ),
        concrete(
            i,
            "optv",
            fn_of(i, &[], vec_of(i, unif(option_of(string())))),
        ),
        concrete(
            i,
            "g2",
            fn_of(i, &[("v", vec_of(i, spec(option_of(string()))))], int()),
        ),
        concrete(i, "boxed", fn_of(i, &[], box2_of(i, spec(string())))),
        concrete(
            i,
            "vec_erase",
            fn_of(
                i,
                &[("v", vec_of(i, spec(t.clone())))],
                vec_of(i, unif(t.clone())),
            ),
        ),
        concrete(
            i,
            "vec_materialize",
            fn_of(i, &[("v", vec_of(i, unif(t.clone())))], vec_of(i, spec(t))),
        ),
    ]
}

const VEC_ARRAY_AT_SPECIALIZED: usize = 0;
const VEC_ARRAY_AT_UNIFORM: usize = 1;
const VEC_ARRAY_CONCRETE: usize = 2;
const VEC_ARRAY_GENERIC: usize = VEC_ARRAY_CONCRETE;
const REVERSE_AT_SPECIALIZED: usize = 0;
const REVERSE_CONCRETE: usize = 1;

// -- Harness ----------------------------------------------------------

fn script_fn(i: &Interner, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern("script")),
        kind: FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script(i, source).expect("parse"),
        )),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
        },
    }
}

struct Checked {
    ret: Ty,
    /// Every extern call the script makes, by callee name, with the
    /// instance the checker settled.
    calls: Vec<(String, usize)>,
    /// The conversions the checker settled at argument positions.
    casts: usize,
}

/// The script's return type and its extern calls, or every error the
/// checker reported.
fn check_with(i: &Interner, reg: TypeRegistry, source: &str) -> Result<Checked, Vec<String>> {
    check_functions(i, reg, externs(i), source)
}

fn check_functions(
    i: &Interner,
    reg: TypeRegistry,
    mut functions: Vec<Function>,
    source: &str,
) -> Result<Checked, Vec<String>> {
    let script = script_fn(i, source);
    let qref = script.qref;
    functions.push(script);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(i, &graph, &ext, &FxHashMap::default(), Freeze::new(reg));
    if inf.has_errors() {
        return Err(inf
            .errors()
            .into_iter()
            .flat_map(|(_, errs)| errs.iter().map(|e| e.display(i).to_string()))
            .collect());
    }
    let outcome = &inf.outcomes[&qref];
    let Ty::Fn { ret, .. } = &outcome.meta().ty else {
        panic!("expected Fn, got {:?}", outcome.meta().ty)
    };
    let resolution = outcome.resolution().expect("complete");
    let casts = resolution.coercion_map.len();
    let mut calls: Vec<(String, usize)> = resolution
        .direct_calls
        .values()
        .filter_map(|callee| match callee {
            Callee::Extern { id, instance } => Some((i.resolve(id.name).to_string(), *instance)),
            _ => None,
        })
        .collect();
    calls.sort();
    Ok(Checked {
        ret: (**ret).clone(),
        calls,
        casts,
    })
}

fn check(i: &Interner, source: &str) -> Result<Checked, Vec<String>> {
    check_with(i, registry(i), source)
}

fn check_casts(i: &Interner, source: &str) -> Result<Checked, Vec<String>> {
    check_with(i, with_casts(i, registry(i)), source)
}

fn ty_of(i: &Interner, source: &str) -> Ty {
    check(i, source).unwrap_or_else(|e| panic!("{e:?}")).ret
}

fn errors_of(i: &Interner, source: &str) -> Vec<String> {
    match check(i, source) {
        Ok(c) => panic!("checked to {:?}", c.ret),
        Err(e) => e,
    }
}

fn spec_ty(ty: Ty) -> TypeArg<acvus_mir::ty::Concrete> {
    TypeArg::specialized(ty)
}

fn unif_ty(ty: Ty) -> TypeArg<acvus_mir::ty::Concrete> {
    TypeArg::uniform(ty)
}

fn instance_of(checked: &Checked, name: &str) -> usize {
    let mut found = checked.calls.iter().filter(|(n, _)| n == name);
    let (_, instance) = found.next().unwrap_or_else(|| panic!("no call to {name}"));
    assert!(found.next().is_none(), "one call to {name}");
    *instance
}

// -- H1: R1 and R2 ----------------------------------------------------

#[test]
fn h1_a_signature_s_representation_is_the_type() {
    let i = Interner::new();
    assert_eq!(ty_of(&i, "f()"), vec_of(&i, spec_ty(Ty::String)));
}

#[test]
fn h1_a_slot_that_does_not_specialize_is_uniform() {
    let i = Interner::new();
    assert_eq!(ty_of(&i, "boxed()"), box2_of(&i, unif_ty(Ty::String)));
}

#[test]
fn h1_the_outermost_and_a_uniform_composite_carry_no_mark() {
    let i = Interner::new();
    assert_eq!(ty_of(&i, "g(f())"), Ty::I64);
    assert_eq!(ty_of(&i, "Some(g(f()))"), Ty::Option(Box::new(Ty::I64)));
}

// -- H2: R3 conversion --------------------------------------------------

#[test]
fn h2_two_representations_at_one_slot_are_an_error_naming_both() {
    let i = Interner::new();
    let errs = errors_of(&i, "g(k())");
    assert!(
        errs.iter()
            .any(|e| e.contains("Vec<#String>") && e.contains("Vec<String>")),
        "{errs:?}"
    );
    let errs = errors_of(&i, "m(f())");
    assert!(
        errs.iter()
            .any(|e| e.contains("Vec<#String>") && e.contains("Vec<String>")),
        "{errs:?}"
    );
}

#[test]
fn h2_a_declared_conversion_bridges_the_two() {
    let i = Interner::new();
    assert_eq!(check_casts(&i, "g(k())").unwrap().ret, Ty::I64);
    assert_eq!(check_casts(&i, "m(f())").unwrap().ret, Ty::I64);
}

// -- H3: the demand reaches the producer ------------------------------

#[test]
fn h3_a_consumer_s_demand_picks_the_producer_s_instance() {
    let i = Interner::new();
    let checked = check(&i, "g(vec_array([\"a\"]))").unwrap();
    assert_eq!(checked.ret, Ty::I64);
    assert_eq!(instance_of(&checked, "vec_array"), VEC_ARRAY_AT_SPECIALIZED);
}

#[test]
fn h3_no_demand_freezes_uniform() {
    let i = Interner::new();
    let checked = check(&i, "vec_array([\"a\"])").unwrap();
    assert_eq!(checked.ret, vec_of(&i, unif_ty(Ty::String)));
    assert_eq!(instance_of(&checked, "vec_array"), VEC_ARRAY_AT_UNIFORM);
}

#[test]
fn h3_the_demand_passes_through_a_generic_signature() {
    let i = Interner::new();
    let checked = check(&i, "g(reverse(vec_array([\"a\"])))").unwrap();
    assert_eq!(checked.ret, Ty::I64);
    assert_eq!(instance_of(&checked, "vec_array"), VEC_ARRAY_AT_SPECIALIZED);
    assert_eq!(instance_of(&checked, "reverse"), REVERSE_AT_SPECIALIZED);
}

#[test]
fn h3_an_element_type_without_an_instance_takes_the_generic_one() {
    let i = Interner::new();
    let checked = check(&i, "vec_array([1])").unwrap();
    assert_eq!(checked.ret, vec_of(&i, unif_ty(Ty::I64)));
    assert_eq!(instance_of(&checked, "vec_array"), VEC_ARRAY_GENERIC);
}

// -- H4: R3 instance ----------------------------------------------------

#[test]
fn h4_a_specialized_binding_without_an_instance_is_an_error() {
    let i = Interner::new();
    let errs = errors_of(&i, "first(f())");
    assert!(
        errs.iter()
            .any(|e| e.contains("no instance") && e.contains("Vec<#String>")),
        "{errs:?}"
    );
}

#[test]
fn h4_a_uniform_binding_takes_the_generic_instance() {
    let i = Interner::new();
    assert_eq!(ty_of(&i, "first(k())"), Ty::String);
}

// -- H5: R1 a composite is laid out whole --------------------------------

#[test]
fn h5_a_composite_inside_a_slot_is_one_representation() {
    let i = Interner::new();
    let option = Ty::Option(Box::new(Ty::String));
    assert_eq!(ty_of(&i, "opt()"), vec_of(&i, spec_ty(option.clone())));
    assert_eq!(ty_of(&i, "optv()"), vec_of(&i, unif_ty(option.clone())));
    assert_eq!(ty_of(&i, "g2(opt())"), Ty::I64);
    let errs = errors_of(&i, "g2(optv())");
    assert!(
        errs.iter()
            .any(|e| e.contains("Vec<#Option<String>>") && e.contains("Vec<Option<String>>")),
        "{errs:?}"
    );
    assert_eq!(ty_of(&i, "first(optv())"), option);
    let errs = errors_of(&i, "first(opt())");
    assert!(errs.iter().any(|e| e.contains("no instance")), "{errs:?}");
}

// -- H6: R2 reopening ---------------------------------------------------

#[test]
fn h6_a_value_at_a_uniform_position_takes_the_next_demand() {
    let i = Interner::new();
    let checked = check(&i, "s = first(k()); g(vec_array([s]))").unwrap();
    assert_eq!(checked.ret, Ty::I64);
    assert_eq!(instance_of(&checked, "vec_array"), VEC_ARRAY_AT_SPECIALIZED);
}

// -- H7: the registry is the solver's ----------------------------------

#[test]
#[should_panic(expected = "unknown UserDefined type")]
fn h7_an_unregistered_type_is_refused_where_it_is_named() {
    let i = Interner::new();
    let reg = TypeRegistry::new();
    let mut sources = acvus_mir::ty::Sources::new();
    let mut solver = acvus_mir::ty::Solver::new(&mut sources, &reg);
    let a = vec_of(
        &i,
        TypeArg::<acvus_mir::ty::Infer>::specialized(TyTerm::String),
    );
    let b = vec_of(&i, TypeArg::<acvus_mir::ty::Infer>::uniform(TyTerm::String));
    let _ = solver.unify(&a, &b);
}

// -- H8: a `Monomorphize` member's registered signature ------------------

/// The arithmetic a member supplies; the bound keeps the generic instance
/// out, so `dot` and `zeros` exist only at `f64`.
trait Float: Copy + Send + Sync + 'static {
    const ZERO: Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl Float for f64 {
    const ZERO: Self = 0.0;

    fn mul_add(self, a: Self, b: Self) -> Self {
        self + a * b
    }
}

#[extern_fn(effect = pure)]
fn dot<T>(a: Vec<T>, b: Vec<T>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    a.iter()
        .zip(&b)
        .fold(T::ZERO, |acc, (x, y)| acc.mul_add(*x, *y))
}

#[extern_fn(effect = pure)]
fn zeros<T>(n: i64) -> Vec<T>
where
    T: Monomorphize<(f64,)> + Float,
{
    vec![T::ZERO; usize::try_from(n).expect("a count is not negative")]
}

fn member_registry() -> Registry<TypesOnly> {
    extern_registry! {
        ns: "t",
        fns: [dot, zeros],
    }
}

/// `zeros<T: Monomorphize<(f64,)>>(i64) -> Vec<T>` feeding
/// `dot<T: Monomorphize<(f64,)>>(Vec<T>, Vec<T>) -> T`, both registered by
/// the macro with `#f64` as their only instance: the consumer's demand
/// picks the producer's instance through the registered signatures, as H3
/// does through hand-built ones, and no conversion sits between them.
#[test]
fn h8_a_member_s_demand_reaches_a_member_producer_through_the_registry() {
    let i = Interner::new();
    let Externs {
        functions, types, ..
    } = Externs::combine(vec![acvus_ext::vec_registry(), member_registry()], &i)
        .expect("registries combine");
    let checked = check_functions(&i, types, functions, "dot(zeros(3), zeros(3))").unwrap();
    assert_eq!(checked.ret, Ty::Float);
    assert_eq!(instance_of(&checked, "dot"), 0);
    let zeros: Vec<usize> = checked
        .calls
        .iter()
        .filter(|(name, _)| name == "zeros")
        .map(|(_, instance)| *instance)
        .collect();
    assert_eq!(zeros, vec![0, 0]);
    assert_eq!(checked.casts, 0);
}
