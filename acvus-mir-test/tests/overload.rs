//! RFC-0043 at the checker's contract: a bare name several namespaces
//! declare is settled by the call's evidence, and the callee the lowering
//! reads is the settled one.
//!
//! Against the standard registries: `min` (`num::min(a, b)`,
//! `iter::min(it)`), `contains` over `string` and `iter`, and `len` over
//! `vec`, `array`, `deque`, and `string`. The fixtures live in namespaces no standard
//! registry declares and use names none declares: `fx_a::probe(&Vec<T>, T)`
//! and `fx_b::probe(Items<T>, T)` are a pair separated by the first
//! argument, `fx_a::size(&Vec<T>)` and `fx_c::size(&Arr<T, N>)` a pair
//! a second use intersects with, `fx_a::apply_any` takes a lambda without
//! saying what its parameter is, and `fx_a::only_vec` is one shape.

use acvus_ext::Items;
use acvus_extern::{
    Arr, Closure, Externs, Registry, Runtime, TypesOnly, Var, extern_fn, extern_registry, kind,
};
use acvus_mir::typeck::CallTarget;
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ir::Callee;
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm, TypeRegistry};
use acvus_utils::{Freeze, Interner};

mod fx_a {
    use super::*;

    #[extern_fn(effect = pure)]
    pub fn probe<T>(c: &Vec<T>, x: T) -> bool
    where
        T: Var<kind::Type>,
    {
        let _ = (c, x);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn size<T>(c: &Vec<T>) -> i64
    where
        T: Var<kind::Type>,
    {
        let _ = c;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn apply_any<A, E, Rt>(f: Closure<(A,), bool, E, Rt>) -> bool
    where
        A: Var<kind::Type>,
        E: Var<kind::Effect>,
        Rt: Runtime,
    {
        let _ = f;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn only_vec<T>(v: &Vec<T>) -> T
    where
        T: Var<kind::Type>,
    {
        let _ = v;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn peek(s: &String) -> i64 {
        let _ = s;
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "fx_a",
            fns: [probe, size, apply_any, only_vec, peek],
        }
    }
}

mod fx_b {
    use super::*;

    #[extern_fn(effect = pure)]
    pub fn probe<T, I, Rt>(it: Items<T, I, Rt>, x: T) -> bool
    where
        T: Var<kind::Type> + acvus_extern::Cross<Rt>,
        I: Var<kind::Identity>,
        Rt: Runtime,
    {
        let _ = (it, x);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn peek(s: &str) -> i64 {
        let _ = s;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn glance(s: &str) -> i64 {
        let _ = s;
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "fx_b",
            fns: [probe, peek, glance],
        }
    }
}

mod fx_c {
    use super::*;

    #[extern_fn(effect = pure)]
    pub fn size<T, N>(c: &Arr<T, N>) -> i64
    where
        T: Var<kind::Type>,
        N: Var<kind::Length>,
    {
        let _ = c;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn glance<T>(c: &Vec<T>) -> i64
    where
        T: Var<kind::Type>,
    {
        let _ = c;
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "fx_c",
            fns: [size, glance],
        }
    }
}

// -- Harness ----------------------------------------------------------------

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
        types: Freeze::new(reg),
        bindings: acvus_mir::graph::Bindings::default(),
        entry: None,
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
    let outcome = &inf.outcomes[&qref];
    let Ty::Fn { ret, .. } = &outcome.meta().ty else {
        panic!("expected Fn, got {:?}", outcome.meta().ty)
    };
    let resolution = outcome.resolution().expect("complete");
    let mut callees: Vec<String> = resolution
        .calls
        .values()
        .filter_map(|target| match target {
            CallTarget::Declared(callee) => Some(callee),
            _ => None,
        })
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
    registries.extend([fx_a::registry(), fx_b::registry(), fx_c::registry()]);
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

const AMBIGUOUS_PROBE: &str = "`probe` is declared by\n  fx_a::probe\n  fx_b::probe";

// -- 1: arity ---------------------------------------------------------------

#[test]
fn min_of_two_values_settles_num_min_by_arity() {
    let i = Interner::new();
    let c = checked(&i, "min(3, 5)");
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(calls(&c, "num::min"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "iter::min"), 0, "{:?}", c.callees);
}

#[test]
fn min_of_an_iterator_settles_iter_min_by_arity() {
    let i = Interner::new();
    let c = checked(&i, "into_iter([3, 5]) | min()");
    assert_eq!(c.ret, Ty::Option(Box::new(Ty::I64)));
    assert_eq!(calls(&c, "iter::min"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "num::min"), 0, "{:?}", c.callees);
}

// -- 2: the first argument's type, against the standard registries -----------

#[test]
fn probe_of_a_lent_vec_settles_the_vec_probe() {
    let i = Interner::new();
    let c = checked(&i, "let v = vec([1, 2]); probe(&v, 2)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_a::probe"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_b::probe"), 0, "{:?}", c.callees);
}

#[test]
fn contains_of_a_lent_string_settles_string_contains() {
    let i = Interner::new();
    let c = checked(&i, r#"let s = "ab"; contains(&s, "b")"#);
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "string::contains"), 1, "{:?}", c.callees);
}

#[test]
fn a_qualified_name_is_that_function() {
    let i = Interner::new();
    let c = checked(&i, r#"let s = "ab"; string::contains(&s, "b")"#);
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "string::contains"), 1, "{:?}", c.callees);
}

#[test]
fn contains_of_an_iterator_settles_iter_contains() {
    let i = Interner::new();
    let c = checked(&i, "into_iter([1]) | contains(1)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "iter::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "string::contains"), 0, "{:?}", c.callees);
}

#[test]
fn len_of_each_container_settles_its_namespace() {
    let i = Interner::new();
    for (source, callee) in [
        ("let v = vec([1]); len(&v)", "vec::len"),
        ("let a = [1]; len(&a)", "array::len"),
        (
            "let d = deque(); push_back(&mut d, 1); len(&d)",
            "deque::len",
        ),
        (r#"let s = "a"; len(&s)"#, "string::len"),
    ] {
        let c = checked(&i, source);
        assert_eq!(
            c.ret,
            Ty::Int(acvus_mir::ty::IntTy::U64),
            "{source}: {:?}",
            c.callees
        );
        assert_eq!(calls(&c, callee), 1, "{source}: {:?}", c.callees);
    }
}

// -- 3: a lambda's parameter, typed by the consumer ----------------------------

#[test]
fn a_lambda_wanted_over_a_lent_vec_settles_the_vec_probe() {
    let i = Interner::new();
    let c = checked(
        &i,
        "into_iter([vec([1])]) | filter(|c| -> probe(c, 1)) | count()",
    );
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(calls(&c, "fx_a::probe"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_b::probe"), 0, "{:?}", c.callees);
}

#[test]
fn a_lambda_wanted_over_an_iterator_settles_the_iterator_probe() {
    let i = Interner::new();
    let c = checked(
        &i,
        "into_iter([into_iter([1])]) | map(|c| -> probe(c, 1)) | count()",
    );
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(calls(&c, "fx_b::probe"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_a::probe"), 0, "{:?}", c.callees);
}

#[test]
fn a_method_call_in_a_lambda_settles_when_the_lambda_is_applied_to_a_lent_vec() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let f = |c, y| -> c.probe(y); let v = vec([1, 2]); f(&v, 2)",
    );
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_a::probe"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_b::probe"), 0, "{:?}", c.callees);
}

#[test]
fn a_method_call_in_a_lambda_settles_when_the_lambda_is_applied_to_a_lent_string() {
    let i = Interner::new();
    let c = checked(
        &i,
        r#"let f = |c, y| -> c.contains(y); let s = "ab"; f(&s, "b")"#,
    );
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "string::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "iter::contains"), 0, "{:?}", c.callees);
}

// -- 4: the intersection of the parameter's uses --------------------------------

#[test]
fn a_second_use_that_only_one_shape_takes_settles_both_names() {
    let i = Interner::new();
    let c = checked(&i, "apply_any(|c| -> probe(c, 1) && size(c) > 0)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_a::probe"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_a::size"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_b::probe"), 0, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_c::size"), 0, "{:?}", c.callees);
}

#[test]
fn a_second_use_with_one_function_settles_the_signature() {
    let i = Interner::new();
    let c = checked(&i, "apply_any(|c| -> probe(c, 1) && only_vec(c) == 1)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_a::probe"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_a::only_vec"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_b::probe"), 0, "{:?}", c.callees);
}

// -- 5: nothing narrows ----------------------------------------------------------

#[test]
fn a_parameter_used_only_in_the_overloaded_call_is_ambiguous_naming_both() {
    let i = Interner::new();
    let errs = errors_of(&i, "apply_any(|c| -> probe(c, 1))");
    assert!(errs.iter().any(|e| e == AMBIGUOUS_PROBE), "{errs:?}");
}

#[test]
fn a_lambda_never_applied_is_ambiguous_naming_both() {
    let i = Interner::new();
    let errs = errors_of(&i, "let f = |c| -> probe(c, 1); true");
    assert!(errs.iter().any(|e| e == AMBIGUOUS_PROBE), "{errs:?}");
}

// -- 6: no candidate -------------------------------------------------------------

#[test]
fn an_argument_no_signature_takes_is_no_matching_function() {
    let i = Interner::new();
    let errs = errors_of(&i, "probe(1, 1)");
    assert!(
        errs.iter()
            .any(|e| e == "no `probe` takes a call of type Fn(i64, i64) -> _"),
        "{errs:?}"
    );
    assert!(!errs.iter().any(|e| e == AMBIGUOUS_PROBE), "{errs:?}");
}

#[test]
fn an_arity_no_signature_has_is_no_matching_function() {
    let i = Interner::new();
    let errs = errors_of(&i, "probe(1)");
    assert!(
        errs.iter()
            .any(|e| e == "no `probe` takes a call of type Fn(i64) -> _"),
        "{errs:?}"
    );
}

// -- 7: an argument a declared conversion takes to a candidate's shape ---------

#[test]
fn a_piped_vec_reaches_iter_contains_through_into_iter() {
    let i = Interner::new();
    let c = checked(&i, "let v = vec([1, 2]); v | into_iter | contains(3)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "iter::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "string::contains"), 0, "{:?}", c.callees);
}

#[test]
fn a_piped_vec_reaches_the_fixture_iterator_probe_through_an_explicit_into_iter() {
    let i = Interner::new();
    let c = checked(&i, "let v = vec([1, 2]); into_iter(v) | probe(3)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_b::probe"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_a::probe"), 0, "{:?}", c.callees);
}

// -- The admission order (RFC-0043, amended) --------------------------------

/// Rule 1: `fx_a::peek` takes the `&String` as it is and `fx_b::peek` would
/// take a view of it, so the view leaves the set at that argument.
#[test]
fn a_candidate_that_takes_the_argument_directly_drops_one_that_would_view_it() {
    let i = Interner::new();
    let c = checked(&i, "let s = \"ab\".to_string(); peek(&s)");
    assert_eq!(
        c.callees,
        vec!["core::to_string".to_string(), "fx_a::peek".to_string()]
    );
}

/// Rule 5: no candidate takes the `&String` directly, `fx_b::glance` takes a
/// view of it, and the view is the checker's — one `as_slice` at the
/// argument, whose instance is the `String`'s own `as_str`.
#[test]
fn a_view_takes_the_call_where_no_candidate_takes_the_argument_directly() {
    let i = Interner::new();
    let c = checked(&i, "let s = \"ab\"; glance(&s)");
    assert_eq!(c.callees, vec!["fx_b::glance".to_string()]);
}

/// Rule 4 and rule 1 at a receiver: a `Vec` receiver is `vec::len`'s own
/// type, and `string::len`'s `&str` is no view of a `Vec`.
#[test]
fn a_container_receiver_takes_its_own_len_and_considers_no_view() {
    let i = Interner::new();
    let c = checked(&i, "let v = [1, 2]; v.len()");
    assert_eq!(c.callees, vec!["array::len".to_string()]);
}

/// Rule 4 at a receiver admitted by the view alone: `string::len` declares
/// `&str`, which no `String` is, and it is the only `len` a `String`
/// receiver reaches.
#[test]
fn a_string_receiver_reaches_len_by_the_view() {
    let i = Interner::new();
    let c = checked(&i, "let s = \"ab\"; s.len()");
    assert_eq!(c.callees, vec!["string::len".to_string()]);
}

/// Rule 2: what `zs[0]` names is a variable where the receiver meets the
/// set, so admission waits for it; when the application names `String`, the
/// re-ask refuses every container `len` and leaves the view.
#[test]
fn an_element_still_a_variable_defers_its_admission_until_the_head_resolves() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let f = |zs| -> zs[0].len(); f([\"a\".to_string(), \"b\".to_string()])",
    );
    assert_eq!(
        c.callees,
        vec![
            "array::as_slice".to_string(),
            "core::to_string".to_string(),
            "core::to_string".to_string(),
            "string::len".to_string()
        ]
    );
}

/// Refused: a set an argument empties opens no decision and names the call
/// as it was written (RFC-0043).
#[test]
fn an_argument_no_candidate_takes_empties_the_set_at_that_argument() {
    let i = Interner::new();
    let errors = errors_of(&i, "peek(1)");
    assert_eq!(errors, vec!["no `peek` takes a call of type Fn(i64) -> _"]);
}

/// Rule 1 at a bare-variable argument, which is the one resolution of the
/// four that admits no view: `k` is a `String` where the call meets the set,
/// no `len` takes a `String` directly, and rule 2 admits a view only from a
/// borrow of a storage — so the set empties and the call is named as written.
#[test]
fn a_bare_variable_receiver_at_a_value_parameter_reaches_no_view() {
    let i = Interner::new();
    let errors = errors_of(&i, "let f = |k| -> len(k); f(\"ab\".to_string())");
    assert_eq!(
        errors,
        vec!["no `len` takes a call of type Fn(String) -> u64"]
    );
}
