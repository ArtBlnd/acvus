//! RFC-0043 at the checker's contract: a bare name several namespaces
//! declare is settled by the call's evidence, and the callee the lowering
//! reads is the settled one.
//!
//! The standard registries already overload `min` (`num::min(a, b)`,
//! `std::min(it)`) and declare `std::contains(it, x)` over an iterator;
//! this file adds `container::contains(c: &C, x: T)` with instances for
//! `Vec<T>` and `Arr<T, N>`, and `t::apply_any`, which takes a lambda
//! without saying what its parameter is.

use acvus_extern::{
    Arr, EffectVar, Externs, Fn1, LenVar, Registry, Runtime, TyVar, TypesOnly, extern_fn,
    extern_registry,
};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ir::Callee;
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm, TypeRegistry};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "container",
        fn contains<C, T>(c: &C, x: T) -> bool
        where
            C: TyVar,
            T: TyVar;
    }
}

#[extern_fn(instance_of = sig::contains, effect = pure)]
fn contains_vec<T>(c: &Vec<T>, x: T) -> bool
where
    T: TyVar,
{
    let _ = (c, x);
    unreachable!("a type-only fixture is never run")
}

#[extern_fn(instance_of = sig::contains, effect = pure)]
fn contains_array<T, N>(c: &Arr<T, N>, x: T) -> bool
where
    T: TyVar,
    N: LenVar,
{
    let _ = (c, x);
    unreachable!("a type-only fixture is never run")
}

#[extern_fn(effect = pure)]
fn apply_any<A, E, Rt>(f: Fn1<A, bool, E, Rt>) -> bool
where
    A: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    let _ = f;
    unreachable!("a type-only fixture is never run")
}

#[extern_fn(effect = pure)]
fn only_vec<T>(v: &Vec<T>) -> T
where
    T: TyVar,
{
    let _ = v;
    unreachable!("a type-only fixture is never run")
}

fn fixture_registry() -> Registry<TypesOnly> {
    extern_registry! {
        ns: "t",
        signatures: [sig::contains],
        fns: [contains_vec, contains_array, apply_any, only_vec],
    }
}

// -- Harness ----------------------------------------------------------------

fn script_fn(i: &Interner, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern("script")),
        kind: FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script_mode(i, source).expect("parse"),
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
    callees: Vec<String>,
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
    let mut callees: Vec<String> = resolution
        .direct_calls
        .values()
        .map(|callee| {
            let id = match callee {
                Callee::Extern { id, .. } | Callee::Direct(id) => id,
                Callee::Indirect(_) => unreachable!("direct_calls holds named callees only"),
            };
            let name = i.resolve(id.name);
            match id.namespace {
                Some(ns) => format!("{}::{name}", i.resolve(ns)),
                None => name.to_string(),
            }
        })
        .collect();
    callees.sort();
    Ok(Checked {
        ret: (**ret).clone(),
        callees,
    })
}

fn check(i: &Interner, source: &str) -> Result<Checked, Vec<String>> {
    let mut registries = acvus_ext::std_registries::<TypesOnly>();
    registries.push(fixture_registry());
    let Externs {
        functions, types, ..
    } = Externs::combine(registries, i).expect("registries combine");
    check_functions(i, types, functions, source)
}

fn checked(i: &Interner, source: &str) -> Checked {
    check(i, source).unwrap_or_else(|e| panic!("{e:?}"))
}

fn errors_of(i: &Interner, source: &str) -> Vec<String> {
    match check(i, source) {
        Ok(c) => panic!("checked to {:?} calling {:?}", c.ret, c.callees),
        Err(e) => e,
    }
}

fn calls(checked: &Checked, callee: &str) -> usize {
    checked
        .callees
        .iter()
        .filter(|c| c.as_str() == callee)
        .count()
}

const AMBIGUOUS_CONTAINS: &str = "`contains` is declared by container::contains and std::contains";
const NO_CONTAINS: &str = "no `contains` takes a call of type";

// -- 1: arity ---------------------------------------------------------------

#[test]
fn min_of_two_values_settles_num_min_by_arity() {
    let i = Interner::new();
    let c = checked(&i, "min(3, 5)");
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(calls(&c, "num::min"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "std::min"), 0, "{:?}", c.callees);
}

#[test]
fn min_of_an_iterator_settles_std_min_by_arity() {
    let i = Interner::new();
    let c = checked(&i, "into_iter([3, 5]) | min()");
    assert_eq!(c.ret, Ty::Option(Box::new(Ty::I64)));
    assert_eq!(calls(&c, "std::min"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "num::min"), 0, "{:?}", c.callees);
}

// -- 2: the first argument's type ---------------------------------------------

#[test]
fn contains_of_a_lent_vec_settles_the_container_signature() {
    let i = Interner::new();
    let c = checked(&i, "let v = vec([1]); contains(&v, 1)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "container::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "std::contains"), 0, "{:?}", c.callees);
}

#[test]
fn contains_of_an_iterator_settles_the_iterator_signature() {
    let i = Interner::new();
    let c = checked(&i, "contains(into_iter([1]), 1)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "std::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "container::contains"), 0, "{:?}", c.callees);
}

// -- 3: a lambda's parameter, typed by the consumer ----------------------------

#[test]
fn a_lambda_wanted_over_a_lent_vec_settles_the_container_signature() {
    let i = Interner::new();
    let c = checked(
        &i,
        "into_iter([vec([1])]) | filter(|c| -> contains(c, 1)) | count()",
    );
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(calls(&c, "container::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "std::contains"), 0, "{:?}", c.callees);
}

#[test]
fn a_lambda_wanted_over_an_iterator_settles_the_iterator_signature() {
    let i = Interner::new();
    let c = checked(
        &i,
        "into_iter([into_iter([1])]) | map(|c| -> contains(c, 1)) | count()",
    );
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(calls(&c, "std::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "container::contains"), 0, "{:?}", c.callees);
}

// -- 4: the intersection of the parameter's uses --------------------------------

/// `C` stays open here, so neither call has a callee (RFC-0040); the
/// settled signature shows as the absence of `AmbiguousFunction`.
#[test]
fn a_second_use_that_only_a_container_takes_settles_without_the_consumer() {
    let i = Interner::new();
    let c = checked(&i, "apply_any(|c| -> contains(c, 1) && len(c) > 0)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "std::contains"), 0, "{:?}", c.callees);
}

#[test]
fn a_second_use_with_one_instance_settles_the_signature_and_the_instance() {
    let i = Interner::new();
    let c = checked(&i, "apply_any(|c| -> contains(c, 1) && only_vec(c) == 1)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "container::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "t::only_vec"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "std::contains"), 0, "{:?}", c.callees);
}

// -- 5: nothing narrows ----------------------------------------------------------

#[test]
fn a_parameter_used_only_in_the_overloaded_call_is_ambiguous_naming_both() {
    let i = Interner::new();
    let errs = errors_of(&i, "apply_any(|c| -> contains(c, 1))");
    assert!(errs.iter().any(|e| e == AMBIGUOUS_CONTAINS), "{errs:?}");
}

#[test]
fn a_lambda_never_applied_is_ambiguous_naming_both() {
    let i = Interner::new();
    let errs = errors_of(&i, "let f = |c| -> contains(c, 1); true");
    assert!(errs.iter().any(|e| e == AMBIGUOUS_CONTAINS), "{errs:?}");
}

// -- 6: no candidate -------------------------------------------------------------

#[test]
fn an_argument_no_signature_takes_is_no_matching_function() {
    let i = Interner::new();
    let errs = errors_of(&i, "contains(1, 1)");
    assert!(errs.iter().any(|e| e.starts_with(NO_CONTAINS)), "{errs:?}");
    assert!(!errs.iter().any(|e| e == AMBIGUOUS_CONTAINS), "{errs:?}");
}

#[test]
fn an_arity_no_signature_has_is_no_matching_function() {
    let i = Interner::new();
    let errs = errors_of(&i, "contains(1)");
    assert!(errs.iter().any(|e| e.starts_with(NO_CONTAINS)), "{errs:?}");
}
