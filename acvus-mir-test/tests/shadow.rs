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
use acvus_mir::typeck::CallTarget;
use acvus_utils::{Freeze, Interner};

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
            flows: acvus_mir::ty::Flows::Every.into(),
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
        types: Freeze::new(types),
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
        panic!("a script is a function: {:?}", outcome.meta().ty)
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
        vec!["`len` is declared by\n  array::len\n  the binding `len`"]
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
    assert_eq!(c.ret, Ty::Int(acvus_mir::ty::IntTy::U64));
    assert_eq!(c.callees, vec!["array::len".to_string()]);
}

#[test]
fn a_binding_made_after_the_call_is_not_in_the_set_at_it() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let q = [1.0, 2.0]; let n = len(&q); let len = |k| -> 7.0; n",
    );
    assert_eq!(c.ret, Ty::Int(acvus_mir::ty::IntTy::U64));
    assert_eq!(c.callees, vec!["array::len".to_string()]);
}

/// `array::len` takes the receiver lent, the binding takes it by value,
/// and per-candidate admission (RFC-0043) leaves both.
#[test]
fn a_method_receiver_two_candidates_take_in_different_modes_is_ambiguous() {
    let i = Interner::new();
    let errors = errors_of(&i, "let q = [1.0, 2.0]; let len = |k| -> 7.0; q.len()");
    assert_eq!(
        errors,
        vec!["`len` is declared by\n  array::len\n  the binding `len`"]
    );
}

/// The modes are read at the method call, not at the call around it.
#[test]
fn a_method_call_of_a_shadowed_name_inside_a_call_of_it_is_ambiguous() {
    let i = Interner::new();
    let errors = errors_of(
        &i,
        "let len = |k| -> k + 7; let q = [1.0, 2.0]; len(q.len())",
    );
    assert_eq!(
        errors,
        vec!["`len` is declared by\n  array::len\n  the binding `len`"]
    );
}

/// Every candidate the receiver's type leaves lends it (RFC-0030).
#[test]
fn a_method_receiver_one_candidate_lends_is_lent() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let mylen = |k| -> k + 7; let q = [1.0, 2.0]; mylen(q.len())",
    );
    assert_eq!(c.ret, Ty::Int(acvus_mir::ty::IntTy::U64));
    assert_eq!(c.callees, vec!["array::len".to_string()]);
    assert!(
        c.fn_params.contains(&vec!["&Array<Float, 2>".to_string()]),
        "the receiver is lent, not moved: {:?}",
        c.fn_params
    );
}

/// Direct first (RFC-0043 rule 1): the binding takes the owned array as it
/// is, `iter::count` takes it only through `into_iter_array`, and the
/// conversion leaves the set at that argument.
#[test]
fn a_binding_that_takes_the_argument_directly_drops_the_signature_that_converts_it() {
    let i = Interner::new();
    let c = checked(&i, "let q = [1.0, 2.0]; let count = |k| -> 7.0; count(q)");
    assert_eq!(c.ret, Ty::Float);
    assert!(
        c.callees.is_empty(),
        "settling on the binding names no function: {:?}",
        c.callees
    );
}

#[test]
fn an_explicit_into_iter_reaches_iter_count_with_no_binding() {
    let i = Interner::new();
    let c = checked(&i, "let q = [1.0, 2.0]; count(into_iter(q))");
    assert_eq!(c.ret, Ty::I64);
    assert_eq!(
        c.callees,
        vec!["iter::count".to_string(), "iter::into_iter".to_string()]
    );
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
        vec!["`len` is declared by\n  array::len\n  the binding `len`"]
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
/// so the binding drops; no declared `len` takes `(i64)` either.
#[test]
fn a_parameter_the_call_fixes_to_a_non_function_drops_from_the_set() {
    let i = Interner::new();
    let errors = errors_of(&i, "let f = |len| -> len(1); f(3)");
    assert_eq!(errors, vec!["no `len` takes a call of type Fn(i64) -> _"]);
}

/// The place is a reference, so the lend is a reborrow and the move is a
/// word copy: both modes see `&Array<i64, 2>`, which is one mode and no
/// ambiguity. The decision then drops the binding, whose `k` the integer
/// bound of `k + 7` holds and a reference does not meet, and `array::len`
/// is alone.
#[test]
fn a_receiver_both_modes_see_as_one_type_is_not_ambiguous() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let len = |k| -> k + 7; let a = [1, 2]; let r = &a; r.len()",
    );
    assert_eq!(c.ret, Ty::Int(acvus_mir::ty::IntTy::U64));
    assert_eq!(c.callees, vec!["array::len".to_string()]);
    assert!(
        c.fn_params.contains(&vec!["i64".to_string()]),
        "the receiver never reached the binding's `k`: {:?}",
        c.fn_params
    );
}

/// The receiver is a lambda parameter the consumer has already fixed to a
/// reference of an open element, so every candidate sees the same `&?`
/// and the five reach the decision under one mode; the element resolving
/// to `Array<i64, 2>` is what leaves `array::len`.
#[test]
fn a_receiver_that_is_a_reference_to_an_open_element_drops_the_binding() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let len = |k| -> k + 7; let a = [[1, 2], [3, 4]]; as_iter(&a) | map(|k| -> k.len() as f64) | sum",
    );
    assert_eq!(c.ret, Ty::Float);
    assert!(
        c.callees.contains(&"array::len".to_string()),
        "{:?}",
        c.callees
    );
}

/// Both candidates see `&Array<i64, 2>`, so the receiver settles on one
/// mode and reports nothing; the binding's `k` is unbounded, so both take
/// the call and `AmbiguousSignature` reports it when the decision fails.
#[test]
fn a_receiver_one_mode_whose_candidates_both_take_the_call_is_ambiguous_at_the_decision() {
    let i = Interner::new();
    let errors = errors_of(
        &i,
        "let len = |k| -> 7.0; let a = [1, 2]; let r = &a; r.len()",
    );
    assert_eq!(
        errors,
        vec!["`len` is declared by\n  array::len\n  the binding `len`"]
    );
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
