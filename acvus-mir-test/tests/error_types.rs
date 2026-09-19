//! A type in an error text is the type as written, a variable nothing
//! resolved closed to `!` (RFC-0043). `<error>` names an `ErrorToken` —
//! a subexpression that already failed to check — and nothing else.

use acvus_extern::{Externs, TypesOnly};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{PolyBuilder, TyTerm};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

/// The errors a script reports.
fn errors(i: &Interner, source: &str) -> Vec<String> {
    let mut pb = PolyBuilder::new();
    let Externs {
        mut functions,
        types,
        ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), i).expect("registries combine");
    functions.push(Function {
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
    });
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(i, &graph, &ext, &FxHashMap::default(), Freeze::new(types));
    inf.errors()
        .into_iter()
        .flat_map(|(_, errs)| errs.iter().map(|e| e.display(i).to_string()))
        .collect()
}

/// The program's reports carry `expected`, and no error token.
fn reports_as_written(source: &str, expected: &str) {
    let i = Interner::new();
    let reported = errors(&i, source);
    assert!(
        reported.iter().any(|e| e == expected),
        "the report states the type as written, `{expected}`: {reported:?}"
    );
    assert!(
        !reported.iter().any(|e| e.contains("<error>")),
        "`<error>` names an ErrorToken and nothing else: {reported:?}"
    );
}

#[test]
fn a_binary_operator_over_two_variables_nothing_resolved_names_them_both() {
    reports_as_written(
        "let f = |k, m| -> { let a = len(k); k < m }; 0",
        "type mismatch in `<`: _ vs _",
    );
}

#[test]
fn an_arithmetic_operator_over_two_variables_nothing_resolved_names_them_both() {
    reports_as_written(
        "let f = |k, m| -> { let a = len(k); k + m }; 0",
        "type mismatch in `+`: _ vs _",
    );
}

#[test]
fn an_enum_pattern_on_an_argument_names_the_enum_it_asked_for() {
    reports_as_written(
        r#"let f = |r| -> { let out = 0; if let Some(v) = r { out = v; }; out }; f(1)"#,
        "type mismatch: expected Option<_>, got i64",
    );
}

#[test]
fn a_tuple_pattern_on_an_argument_names_its_arity() {
    reports_as_written(
        r#"let f = |r| -> { let out = 0; if let (a, b) = r { out = 2; }; out }; f(1)"#,
        "type mismatch: expected (_, _), got i64",
    );
}

#[test]
fn an_object_pattern_on_an_argument_names_the_field_it_asked_for() {
    reports_as_written(
        r#"let f = |r| -> { let out = 0; if let { a, } = r { out = 2; }; out }; f(1)"#,
        "type mismatch: expected {a: _}, got i64",
    );
}

#[test]
fn a_store_through_a_shared_reference_names_the_reference() {
    reports_as_written(
        "let q = 1; let r = &q; *r = 2; 0",
        "cannot store through &_: not a `&mut`",
    );
}

#[test]
fn a_branch_that_does_not_meet_the_other_names_the_branch_type() {
    reports_as_written(
        "let f = |k, m| -> { let a = len(k); let b = if true { k } else { 1 }; b }; 0",
        "type mismatch: expected _, got i64",
    );
}

#[test]
fn a_return_a_conversion_could_not_reach_names_the_type_it_asked_for() {
    reports_as_written(
        "let f = |k| -> { let a = len(k); k? }; f(1)",
        "type mismatch: expected Result<Result<_, _>, _>, got i64",
    );
}

#[test]
fn an_argument_no_candidate_takes_names_the_argument_as_written() {
    reports_as_written(
        "let f = |k| -> len(k); f(1)",
        "type i64 is outside the declared bound one of &Vec<#?'0>, &str, &Deque<#?'0>, \
         &Array<'0, '0>",
    );
}
