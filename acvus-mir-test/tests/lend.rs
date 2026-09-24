//! What a reference to a place names, stated at the checker's contract:
//! the type a lambda's lent parameter is checked at, and the error where
//! the place already holds a reference. A borrow of a reference is a
//! reborrow of what it names (RFC-0029); a lambda captures one as the
//! word it is (RFC-0064 rule 5). A test that fails is a finding, kept
//! as it fails.

use acvus_extern::{Externs, TypesOnly};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{LenTerm, Mutability, PolyBuilder, Ty, TyTerm, TypeArg};
use acvus_utils::{Freeze, Interner};

fn script_fn(i: &Interner, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern("script")),
        kind: FnKind::Local(
            ParsedAst::Script(acvus_ast::parse_script(i, source).expect("parse")),
            acvus_mir::graph::Inputs::FromReads,
        ),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }
}

fn recorded_types(i: &Interner, source: &str) -> Result<Vec<Ty>, Vec<String>> {
    let Externs {
        mut functions,
        types,
        ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), i).expect("registries combine");
    let script = script_fn(i, source);
    let qref = script.qref;
    functions.push(script);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(types),
        bindings: acvus_mir::graph::Bindings::default(),
        entries: Vec::new(),
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(i, &graph, &ext);
    if inf.has_errors() {
        return Err(inf
            .errors()
            .into_iter()
            .flat_map(|(_, errs)| errs.iter().map(|e| e.display(i).to_string()))
            .collect());
    }
    let resolution = inf.outcomes[&qref].resolution().expect("complete");
    Ok(resolution.type_map.values().cloned().collect())
}

fn checked(i: &Interner, source: &str) -> Vec<Ty> {
    recorded_types(i, source).unwrap_or_else(|e| panic!("{}", e.join("\n")))
}

fn shared_ref(ty: Ty) -> Ty {
    Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(ty)))
}

fn array_of_two_floats() -> Ty {
    Ty::Array(Box::new(Ty::Float), LenTerm::Known(2))
}

/// RFC-0029.
fn is_double_reference(ty: &Ty) -> bool {
    matches!(ty, Ty::Ref(_, arg) if matches!(*arg.ty(), Ty::Ref(..)))
}

fn distinct_lent_parameter_types(types: &[Ty]) -> Vec<Ty> {
    let mut lent: Vec<Ty> = types
        .iter()
        .filter_map(|ty| match ty {
            Ty::Fn { params, .. } => Some(params.iter().map(|p| p.ty.clone())),
            _ => None,
        })
        .flatten()
        .filter(|ty| matches!(ty, Ty::Ref(..)))
        .collect();
    lent.sort_by_key(|ty| format!("{ty:?}"));
    lent.dedup();
    lent
}

const OWNED: &str = "let q = [1.0, 1.0]; ";

const BY_METHOD: &str = "let f = |k| -> k.as_iter() | map(|x| -> *x) | sum; f(&q)";
const BY_CALL: &str = "let f = |k| -> as_iter(k) | map(|x| -> *x) | sum; f(&q)";

#[test]
fn a_method_receiver_that_is_a_lent_parameter_is_reborrowed_not_doubled() {
    let i = Interner::new();
    let types = checked(&i, &format!("{OWNED}{BY_METHOD}"));
    assert_eq!(
        distinct_lent_parameter_types(&types),
        vec![shared_ref(array_of_two_floats()), shared_ref(Ty::Float)],
        "`k` is lent once as `&Array`, and `map`'s element once as `&Float`"
    );
    assert!(
        !types.iter().any(is_double_reference),
        "no `&&T` exists (RFC-0029): {types:?}"
    );
}

#[test]
fn a_lent_parameter_passed_to_a_bare_call_is_reborrowed_not_doubled() {
    let i = Interner::new();
    let types = checked(&i, &format!("{OWNED}{BY_CALL}"));
    assert_eq!(
        distinct_lent_parameter_types(&types),
        vec![shared_ref(array_of_two_floats()), shared_ref(Ty::Float)],
        "a bare call lends exactly what the method call does"
    );
    assert!(
        !types.iter().any(is_double_reference),
        "no `&&T` exists (RFC-0029): {types:?}"
    );
}

/// RFC-0064 rule 5 admits the capture, and RFC-0029 still holds of it:
/// the inner lambda's capture type is the reference `a` already is, not a
/// reference to it.
#[test]
fn an_inner_lambda_capturing_a_lent_parameter_captures_it_without_doubling() {
    let i = Interner::new();
    let types = checked(
        &i,
        &format!("{OWNED}let f = |a| -> range(0, 2) | map(|i| -> a[0]) | sum; f(&q)"),
    );
    let captured: Vec<&Vec<Ty>> = types
        .iter()
        .filter_map(|ty| match ty {
            Ty::Fn { captures, .. } if !captures.is_empty() => Some(captures),
            _ => None,
        })
        .collect();
    assert!(
        captured
            .iter()
            .any(|of_one| *of_one == &[shared_ref(array_of_two_floats())]),
        "the inner lambda captures `&Array<Float, 2>`: {captured:?}"
    );
    assert!(
        !types.iter().any(is_double_reference),
        "no `&&T` exists (RFC-0029): {types:?}"
    );
}

#[test]
fn the_same_lambda_over_an_owned_argument_captures_the_value_and_checks() {
    let i = Interner::new();
    let types = checked(&i, &format!("{OWNED}let f = |a| -> a[0] + 0.0; f(q)"));
    assert!(
        !types.iter().any(is_double_reference),
        "no `&&T` exists (RFC-0029): {types:?}"
    );
}

#[test]
fn a_lambda_expecting_a_reference_called_with_an_owned_value_names_both_types() {
    let i = Interner::new();
    let errors = recorded_types(
        &i,
        "let q = [1.0, 1.0]; let f = |k| -> k[0] + 0.0; let r = &q; f(r); f(q)",
    )
    .expect_err("`q` is not the `&Array<Float, 2>` the first call fixed `k` to");
    assert!(
        errors.iter().any(|e| e.contains("Array<Float, 2>")),
        "the message names the argument's type: {errors:?}"
    );
    assert!(
        !errors.iter().any(|e| e.contains("<error>")),
        "a refused argument is not a cascade: {errors:?}"
    );
}

#[test]
fn a_lambda_never_called_closes_its_lend_on_a_plain_reference() {
    let i = Interner::new();
    let refusals = recorded_types(&i, "let f = |k| -> k.as_iter() | map(|x| -> *x) | sum; 0")
        .expect_err("nothing decides the element type of `k`, so no instance is chosen");
    assert!(
        refusals.iter().all(|r| r.contains("cannot infer type")),
        "{refusals:?}"
    );
    assert!(
        refusals.iter().all(|r| !r.contains("&&")),
        "the lend closed on a plain reference, and no `&&T` exists (RFC-0029): {refusals:?}"
    );
}
