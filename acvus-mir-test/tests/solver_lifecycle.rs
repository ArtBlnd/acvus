//! The solver separates equality from decision (scratchpad/tobe/solver.md):
//! unification is the join of the type lattice, a decision is a set of
//! admissible answers that settles when one remains, and a body is
//! checked, then queried, then solved once.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ir::{Callee, CastKind};
use acvus_mir::solver::{Answer, Conversion, Decision, InstanceChoice, InstanceKind};
use acvus_mir::ty::{
    CastRule, Effect, Instances, LenTerm, ObjectTy, ParamTerm, Poly, PolyBuilder, PolyTy, Repr,
    Scheme, Solver, Sources, Ty, TyTerm, TyVarBound, TypeArg, TypeRegistry, UserDefinedDecl,
};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

// -- Types ------------------------------------------------------------

fn vec_ref(i: &Interner) -> QualifiedRef {
    QualifiedRef::root(i.intern("Vec"))
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

fn registry(i: &Interner) -> TypeRegistry {
    let mut reg = TypeRegistry::new();
    reg.register(UserDefinedDecl {
        qref: vec_ref(i),
        type_params: vec![TyVarBound::Any],
        effect_params: 0,
        identity_params: 0,
        specializable: vec![true],
    });
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    reg.register_cast(CastRule {
        from: vec_of(i, TypeArg::uniform(t.clone())),
        to: vec_of(i, TypeArg::specialized(t)),
        fn_ref: QualifiedRef::root(i.intern("vec_materialize")),
    });
    reg
}

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

/// `k() -> Vec<String>`, `g(Vec<#String>) -> Int`, `vec_materialize`, and
/// `vec_array<T, N>(Array<T, N>) -> Vec<T>` at `#String`, `String`, generic.
fn externs(i: &Interner) -> Vec<Function> {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    let n = pb.fresh_len_var();
    let TyTerm::Var(tv) = t else {
        panic!("a type variable")
    };
    let array_of = |elem: PolyTy| TyTerm::Array(Box::new(elem), n.clone());
    let string = || TyTerm::String;
    let mut vec_array_concrete = vec![TyTerm::Unit; VEC_ARRAY_CONCRETE];
    vec_array_concrete[VEC_ARRAY_AT_SPECIALIZED] = fn_of(
        i,
        &[("items", array_of(string()))],
        vec_of(i, TypeArg::specialized(string())),
    );
    vec_array_concrete[VEC_ARRAY_AT_UNIFORM] = fn_of(
        i,
        &[("items", array_of(string()))],
        vec_of(i, TypeArg::uniform(string())),
    );
    vec![
        extern_fn(
            i,
            "k",
            fn_of(i, &[], vec_of(i, TypeArg::uniform(string()))),
            Instances::default(),
        ),
        extern_fn(
            i,
            "g",
            fn_of(
                i,
                &[("v", vec_of(i, TypeArg::specialized(string())))],
                TyTerm::I64,
            ),
            Instances::default(),
        ),
        extern_fn(
            i,
            "vec_materialize",
            fn_of(
                i,
                &[("v", vec_of(i, TypeArg::uniform(t.clone())))],
                vec_of(i, TypeArg::specialized(t.clone())),
            ),
            Instances::default(),
        ),
        extern_fn(
            i,
            "vec_array",
            fn_of(
                i,
                &[("items", array_of(t.clone()))],
                vec_of(i, TypeArg::new(Repr::Var(tv), t)),
            ),
            Instances {
                concrete: vec_array_concrete
                    .into_iter()
                    .map(acvus_mir::ty::InstanceSig::any_task)
                    .collect(),
                generic: true,
            },
        ),
    ]
}

const VEC_ARRAY_AT_SPECIALIZED: usize = 0;
const VEC_ARRAY_AT_UNIFORM: usize = 1;
const VEC_ARRAY_CONCRETE: usize = 2;

// -- Harness ----------------------------------------------------------

struct Checked {
    ret: Ty,
    calls: Vec<(String, usize)>,
    coercions: Vec<String>,
}

fn check(i: &Interner, source: &str) -> Result<Checked, Vec<String>> {
    let mut pb = PolyBuilder::new();
    let script = Function {
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
    };
    let qref = script.qref;
    let mut functions = externs(i);
    functions.push(script);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(
        i,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(registry(i)),
    );
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
    let mut calls: Vec<(String, usize)> = resolution
        .direct_calls
        .values()
        .filter_map(|callee| match callee {
            Callee::Extern { id, instance } => Some((i.resolve(id.name).to_string(), *instance)),
            _ => None,
        })
        .collect();
    calls.sort();
    let mut coercions: Vec<String> = resolution
        .coercion_map
        .iter()
        .map(|(_, kind)| match kind {
            CastKind::Extern { fn_ref, .. } => i.resolve(fn_ref.name).to_string(),
            CastKind::ThroughRef { cast, back, .. } => format!(
                "&{}/{}",
                i.resolve(cast.fn_ref.name),
                i.resolve(back.fn_ref.name)
            ),
            CastKind::Slice { as_slice, .. } => {
                format!("[{}]", i.resolve(as_slice.fn_ref.name))
            }
        })
        .collect();
    coercions.sort();
    Ok(Checked {
        ret: (**ret).clone(),
        calls,
        coercions,
    })
}

fn ok(i: &Interner, source: &str) -> Checked {
    check(i, source).unwrap_or_else(|e| panic!("{e:?}"))
}

fn errors(i: &Interner, source: &str) -> Vec<String> {
    match check(i, source) {
        Ok(c) => panic!("checked to {:?}", c.ret),
        Err(e) => e,
    }
}

// -- S1: a conversion at a ground argument ---------------------------

#[test]
fn s1_a_conversion_between_two_ground_types_is_recorded_at_the_argument() {
    let i = Interner::new();
    let checked = ok(&i, "g(k())");
    assert_eq!(checked.ret, Ty::I64);
    assert_eq!(checked.coercions, vec!["vec_materialize".to_string()]);
}

// -- S2: no lub; structural types join by union --------------------------

#[test]
fn s2_two_objects_join_to_the_union_of_their_fields() {
    let i = Interner::new();
    let checked = ok(&i, "[{ a: 1, }, { b: \"s\", }]");
    let fields: FxHashMap<_, _> = [(i.intern("a"), Ty::I64), (i.intern("b"), Ty::String)]
        .into_iter()
        .collect();
    assert_eq!(
        checked.ret,
        Ty::Array(
            Box::new(Ty::Object(ObjectTy::written(fields))),
            LenTerm::Known(2)
        )
    );
}

#[test]
fn s2_a_field_read_is_a_join_with_a_partial_object() {
    let i = Interner::new();
    let checked = ok(&i, "let f = |o| -> o.name; f({ name: \"n\", age: 1, })");
    assert_eq!(checked.ret, Ty::String);
}

// -- S3: one settlement ---------------------------------------------------

#[test]
fn s3_dependent_decisions_settle_in_one_call() {
    let i = Interner::new();
    let reg = registry(&i);
    let mut sources = Sources::new();
    let mut solver = Solver::new(&mut sources, &reg);

    let vec_array = externs(&i)
        .into_iter()
        .find(|f| f.qref == QualifiedRef::root(i.intern("vec_array")))
        .expect("declared");
    let FnKind::Extern { instances, .. } = vec_array.kind else {
        panic!("an extern")
    };
    let instantiated = solver.instantiate_scheme(&Scheme {
        ty: vec_array.ty,
        bounds: vec![],
        instances: Some(instances),
    });
    let Some(InstanceChoice::Decided(instance)) = instantiated.instance else {
        panic!("a decision among instances")
    };
    let TyTerm::Fn { params, ret, .. } = instantiated.ty else {
        panic!("a function")
    };

    let array = TyTerm::Array(Box::new(TyTerm::String), LenTerm::Known(1));
    solver
        .unify(&params[0].ty, &array)
        .expect("the argument joins");
    let demand = vec_of(&i, TypeArg::specialized(TyTerm::String));
    let conversion = solver.decide(Decision::conversion(&ret, &demand));

    let unsettled = solver.settle();
    assert!(unsettled.is_empty(), "{unsettled:?}");
    assert_eq!(
        solver.answer(instance),
        Some(Answer::Instance(InstanceKind::Extern(
            VEC_ARRAY_AT_SPECIALIZED
        )))
    );
    assert_eq!(
        solver.answer(conversion),
        Some(Answer::Conversion(Conversion::Identity))
    );
    assert_eq!(
        solver.freeze_ty(&ret).expect("solved"),
        vec_of(&i, TypeArg::specialized(Ty::String))
    );
}

// -- S4: what remains open is named ------------------------------------------

#[test]
fn s4_a_conversion_that_does_not_exist_names_both_types_at_the_argument() {
    let i = Interner::new();
    let errs = errors(&i, "g([1])");
    assert!(
        errs.iter()
            .any(|e| e.contains("Vec<#String>") && e.contains("Array<i64, 1>")),
        "{errs:?}"
    );
}

// -- S5: RFC-0027 without a mid-body settlement -----------------------------

#[test]
fn s5_a_lambda_after_the_argument_sees_the_element_type_the_instance_fixed() {
    let i = Interner::new();
    let checked = ok(&i, "vec_array([\"a\"]) ; let f = |x| -> x + 1; f(1)");
    assert_eq!(checked.ret, Ty::I64);
    assert_eq!(
        checked.calls,
        vec![("vec_array".to_string(), VEC_ARRAY_AT_UNIFORM)]
    );
}

// -- S6: an object grows by a field store -----------------------------------

/// `x.b = 0` grows `x` to `{a, b}` on every path; whether `b` is
/// initialized where `f` reads it is the definite-assignment check's
/// question (validate/init_check.rs), not the type checker's.
#[test]
fn s6_a_field_store_grows_the_object_for_every_use() {
    let i = Interner::new();
    let ab = TyTerm::Object(ObjectTy::written(
        [(i.intern("a"), TyTerm::I64), (i.intern("b"), TyTerm::I64)]
            .into_iter()
            .collect(),
    ));
    let fab = extern_fn(
        &i,
        "fab",
        fn_of(&i, &[("v", ab)], TyTerm::I64),
        Instances::default(),
    );
    let mut pb = PolyBuilder::new();
    let script = Function {
        qref: QualifiedRef::root(i.intern("script")),
        kind: FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script(
                &i,
                "let x = { a: 1, }; let y = if true { x.b = 0; fab(x) } else { fab(x) }; y",
            )
            .expect("parse"),
        )),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
        },
    };
    let qref = script.qref;
    let graph = CompilationGraph {
        functions: Freeze::new(vec![fab, script]),
        contexts: Freeze::new(vec![]),
    };
    let ext = extract::extract(&i, &graph);
    let inf = infer::infer(
        &i,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(registry(&i)),
    );
    let errs: Vec<String> = inf
        .errors()
        .into_iter()
        .flat_map(|(_, errs)| errs.iter().map(|e| e.display(&i).to_string()))
        .collect();
    assert!(errs.is_empty(), "{errs:?}");
    let Ty::Fn { ret, .. } = &inf.outcomes[&qref].meta().ty else {
        panic!("a function")
    };
    assert_eq!(**ret, Ty::I64);
}
