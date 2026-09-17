//! A `let` binding of a function is one more signature of its bare name
//! (RFC-0043), decided by the call's evidence as a namespace's signature
//! is: where both take the call it is ambiguous, where only the binding
//! takes it the binding is called.
//!
//! Against the standard registries, where `len` is declared by `array`,
//! `vec`, `deque` and `string`, and `count` by `iter`.

use acvus_extern::{Externs, TypesOnly};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ir::Callee;
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

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
    fn_params: Vec<Vec<String>>,
}

fn check(i: &Interner, source: &str) -> Result<Checked, Vec<String>> {
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
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(i, &graph, &ext, &FxHashMap::default(), Freeze::new(types));
    if inf.has_errors() {
        return Err(inf
            .errors()
            .into_iter()
            .flat_map(|(_, errs)| errs.iter().map(|e| e.display(i).to_string()))
            .collect());
    }
    let outcome = &inf.outcomes[&qref];
    let Ty::Fn { ret, .. } = &outcome.meta().ty else {
        panic!("a script is a function: {:?}", outcome.meta().ty)
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
    let mut fn_params: Vec<Vec<String>> = resolution
        .type_map
        .values()
        .filter_map(|ty| match ty {
            Ty::Fn { params, .. } => {
                Some(params.iter().map(|p| p.ty.display(i).to_string()).collect())
            }
            _ => None,
        })
        .collect();
    fn_params.sort();
    fn_params.dedup();
    Ok(Checked {
        ret: (**ret).clone(),
        callees,
        fn_params,
    })
}

fn checked(i: &Interner, source: &str) -> Checked {
    check(i, source).unwrap_or_else(|e| panic!("{}", e.join("\n")))
}

fn errors_of(i: &Interner, source: &str) -> Vec<String> {
    match check(i, source) {
        Ok(c) => panic!("checked to {:?} calling {:?}", c.ret, c.callees),
        Err(e) => e,
    }
}

#[test]
fn a_binding_and_an_extern_that_both_take_the_call_are_ambiguous() {
    let i = Interner::new();
    let errors = errors_of(&i, "let q = [1.0, 2.0]; let len = |k| -> 7.0; len(&q)");
    assert_eq!(
        errors,
        vec!["`len` is declared by array::len and the binding `len`"]
    );
}

#[test]
fn a_binding_alone_takes_a_call_no_declared_len_takes() {
    let i = Interner::new();
    let c = checked(&i, "let len = |k| -> k + 7; len(1)");
    assert_eq!(c.ret, Ty::I64);
    assert!(
        c.callees.is_empty(),
        "the binding is called, not a namespace's `len`: {:?}",
        c.callees
    );
    assert!(
        c.fn_params.contains(&vec!["i64".to_string()]),
        "the binding's parameter is the argument's type: {:?}",
        c.fn_params
    );
}

/// `iter::count` takes an `Iterator` by value. The only declared cast that
/// reaches `Iterator` from an array is `into_iter_array: Arr<T, N> ->
/// Iter<T, E, I, Rt>`, whose source is an owned array, so no rule admits
/// `&Array<Float, 2>` and the binding alone takes the call.
#[test]
fn a_reference_no_cast_reaches_an_iterator_from_leaves_the_binding_alone() {
    let i = Interner::new();
    let c = checked(&i, "let q = [1.0, 2.0]; let count = |k| -> 7.0; count(&q)");
    assert_eq!(c.ret, Ty::Float);
    assert!(
        c.callees.is_empty(),
        "the binding is called, not `iter::count`: {:?}",
        c.callees
    );
    assert!(
        c.fn_params.contains(&vec!["&Array<Float, 2>".to_string()]),
        "{:?}",
        c.fn_params
    );
}

#[test]
fn a_binding_no_namespace_declares_is_called_as_before() {
    let i = Interner::new();
    let c = checked(&i, "let mylen = |k| -> k + 7; mylen(1)");
    assert_eq!(c.ret, Ty::I64);
    assert!(c.callees.is_empty(), "{:?}", c.callees);
}

#[test]
fn a_binding_that_is_not_a_function_does_not_join_the_set() {
    let i = Interner::new();
    let c = checked(&i, "let len = 3; let q = [1.0, 2.0]; len(&q)");
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(c.callees, vec!["array::len".to_string()]);
}

#[test]
fn a_binding_made_after_the_call_is_not_in_the_set_at_it() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let q = [1.0, 2.0]; let n = len(&q); let len = |k| -> 7.0; n",
    );
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(c.callees, vec!["array::len".to_string()]);
}

/// RFC-0030 lends the receiver only where every candidate's first
/// parameter is a reference of one mutability. The binding's is a
/// variable, so the receiver is passed by value, which every declared
/// `len` refuses.
#[test]
fn a_method_receiver_a_binding_may_take_is_passed_by_value() {
    let i = Interner::new();
    let c = checked(&i, "let q = [1.0, 2.0]; let len = |k| -> 7.0; q.len()");
    assert_eq!(c.ret, Ty::Float);
    assert!(
        c.callees.is_empty(),
        "the binding is called, not a namespace's `len`: {:?}",
        c.callees
    );
    assert!(
        c.fn_params.contains(&vec!["Array<Float, 2>".to_string()]),
        "the receiver arrives owned: {:?}",
        c.fn_params
    );
}

/// Taking the call includes taking an argument through one declared
/// conversion: `iter::count` takes the owned array by `into_iter_array`,
/// the binding takes it as it is, and both remain.
#[test]
fn a_binding_and_a_signature_that_converts_the_argument_are_ambiguous() {
    let i = Interner::new();
    let errors = errors_of(&i, "let q = [1.0, 2.0]; let count = |k| -> 7.0; count(q)");
    assert_eq!(
        errors,
        vec!["`count` is declared by iter::count and the binding `count`"]
    );
}

#[test]
fn the_conversion_that_reaches_iter_count_still_runs_with_no_binding() {
    let i = Interner::new();
    let c = checked(&i, "let q = [1.0, 2.0]; count(q)");
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(c.callees, vec!["iter::count".to_string()]);
}

/// A lambda's parameter is such a binding: nothing has fixed its head
/// where the body's call stands, and the call is what fixes it. No
/// declared `len` takes `(i64)`, so the parameter is alone and settling
/// on it gives it the call's function type.
#[test]
fn a_binding_whose_head_is_still_a_variable_joins_the_set() {
    let i = Interner::new();
    let c = checked(&i, "let apply = |len| -> len(1); apply(|k| -> k + 7)");
    assert_eq!(c.ret, Ty::I64);
    assert!(
        c.callees.is_empty(),
        "the parameter is called, not a namespace's `len`: {:?}",
        c.callees
    );
}

#[test]
fn a_parameter_and_an_extern_that_both_take_the_call_are_ambiguous() {
    let i = Interner::new();
    let errors = errors_of(
        &i,
        "let q = [1.0, 2.0]; let f = |len| -> len(&q); f(|k| -> 7.0)",
    );
    assert_eq!(
        errors,
        vec!["`len` is declared by array::len and the binding `len`"]
    );
}

#[test]
fn a_parameter_alone_where_no_rule_reaches_the_extern_is_the_callee() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let q = [1.0, 2.0]; let f = |count| -> count(&q); f(|k| -> 7.0)",
    );
    assert_eq!(c.ret, Ty::Float);
    assert!(c.callees.is_empty(), "{:?}", c.callees);
}

/// The argument fixes the parameter to an `i64`, which is not callable,
/// so the binding drops; no declared `len` takes `(i64)` either. The
/// call's return is still open where the failure is reported, and
/// `freeze_or_error` refuses an open variable, so the type shown is
/// `<error>` — as it is at every `NoMatchingFunction`.
#[test]
fn a_parameter_the_call_fixes_to_a_non_function_drops_from_the_set() {
    let i = Interner::new();
    let errors = errors_of(&i, "let f = |len| -> len(1); f(3)");
    assert_eq!(errors, vec!["no `len` takes a call of type <error>"]);
}

#[test]
fn a_parameter_no_namespace_declares_is_the_callee() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let q = [1.0, 2.0]; let f = |g| -> g(&q); f(|k| -> 7.0)",
    );
    assert_eq!(c.ret, Ty::Float);
    assert!(c.callees.is_empty(), "{:?}", c.callees);
}
