//! A captured name of word type is a copy of a value the closure owns and
//! a captured name of any other type is a borrow of one (RFC-0018), at the
//! checker's contract and at the IR the lowering writes. A captured
//! reference is a word — the `Kind::Ref` word itself (RFC-0064 rule 5).
//! A test that fails is a finding, kept as it fails.

use acvus_extern::{Externs, TypesOnly};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{LenTerm, Mutability, PolyBuilder, Ty, TyTerm, TypeArg};
use acvus_mir_test::compile_script_mode_optimized;
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

#[derive(Debug)]
struct Checked {
    ret: Ty,
    captures_by_lambda: Vec<Vec<Ty>>,
}

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
    let outcome = &inf.outcomes[&qref];
    let Ty::Fn { ret, .. } = &outcome.meta().ty else {
        panic!("a script is a function: {:?}", outcome.meta().ty)
    };
    let resolution = outcome.resolution().expect("complete");
    let mut captures_by_lambda: Vec<Vec<Ty>> = resolution
        .type_map
        .iter()
        .filter_map(|(_, ty)| match ty {
            Ty::Fn { captures, .. } if !captures.is_empty() => Some(captures.clone()),
            _ => None,
        })
        .collect();
    captures_by_lambda.sort_by_key(|c| format!("{c:?}"));
    Ok(Checked {
        ret: (**ret).clone(),
        captures_by_lambda,
    })
}

fn checked(i: &Interner, source: &str) -> Checked {
    check(i, source).unwrap_or_else(|e| panic!("{}", e.join("\n")))
}

#[test]
fn a_let_bound_lambda_is_called_from_inside_another_lambda() {
    let i = Interner::new();
    let c = checked(&i, "let f = |x| -> x + 1.0; let g = |t| -> f(1.0); g(0)");
    assert_eq!(c.ret, Ty::Float);
}

#[test]
fn a_let_bound_lambda_is_called_from_a_lambda_an_extern_receives() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let f = |x| -> x + 1.0; range(0, 2) | map(|t| -> f(1.0)) | sum",
    );
    assert_eq!(c.ret, Ty::Float);
}

const TWO_LEVEL_WORD: &str = "let h = 0.5; \
     range(0, 2) | map(|j| -> range(0, 2) | map(|t| -> h * t as f64) | sum) | sum";

const TWO_LEVEL_LARGE: &str = "let w = [0.5, 0.5]; \
     as_iter(&@values) | map(|row| -> as_iter(row) | map(|x| -> w[0] * *x) | sum) | sum";

#[test]
fn a_two_level_capture_records_the_owned_type_at_both_levels() {
    let i = Interner::new();
    let c = checked(&i, TWO_LEVEL_WORD);
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(
        c.captures_by_lambda,
        vec![vec![Ty::Float], vec![Ty::Float]],
        "each lambda captures the value, never the `&T` the name reads as"
    );
}

#[test]
fn a_two_level_capture_of_a_large_value_is_a_move_out_of_the_capture() {
    let i = Interner::new();
    let errors = check(&i, TWO_LEVEL_LARGE).expect_err("the inner lambda moves `w`");
    assert!(
        errors.iter().any(|e| e
            == "cannot move `w` out of a closure's capture (type Array<Float, 2>); \
                act through the reference, or clone it"),
        "{errors:?}"
    );
}

#[test]
fn a_two_level_capture_of_a_word_lowers_to_a_closure_that_owns_its_capture() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(&i, TWO_LEVEL_WORD, &FxHashMap::default());
    assert!(ir.is_ok(), "{}", ir.unwrap_err());
}

#[test]
fn a_lambda_capturing_a_lent_parameter_captures_the_reference_itself() {
    let i = Interner::new();
    let checked = check(
        &i,
        "let a = [[0.5, 0.5]]; \
         as_iter(&a) | map(|k| -> range(0, 2) | map(|t| -> k[0] + t as f64) | sum) | sum",
    )
    .unwrap_or_else(|errors| panic!("{errors:?}"));
    let captured = Ty::Ref(
        Mutability::Shared,
        Box::new(TypeArg::uniform(Ty::Array(
            Box::new(Ty::Float),
            LenTerm::Known(2),
        ))),
    );
    assert!(
        checked
            .captures_by_lambda
            .iter()
            .any(|of_one| of_one == &[captured.clone()]),
        "{checked:?}"
    );
}

#[test]
fn a_captured_lambda_passed_by_value_is_still_a_reference_in_a_value_position() {
    let i = Interner::new();
    let errors = check(
        &i,
        "let f = |x| -> x + 1.0; \
         range(0, 2) | map(|t| -> range(0, 2) | map(f) | sum) | sum",
    )
    .expect_err("a `&Fn` does not meet an `Fn` parameter");
    assert!(
        errors.iter().any(|e| e.contains("&Fn(Float) -> Float")),
        "{errors:?}"
    );
}

const LEN_BINDING: &str = "let len = |k| -> k + 1; ";

const LENT_ARRAY: &str = "let a = [[1, 2], [3, 4]]; ";

#[test]
fn a_qualified_call_of_a_name_a_binding_also_has_captures_nothing() {
    let i = Interner::new();
    let source =
        format!("{LEN_BINDING}{LENT_ARRAY}as_iter(&a) | map(|k| -> array::len(k) as f64) | sum");
    let c = checked(&i, &source);
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(
        c.captures_by_lambda,
        Vec::<Vec<Ty>>::new(),
        "a qualified name is not a read of the binding"
    );
    let ir = compile_script_mode_optimized(&i, &source, &FxHashMap::default());
    assert!(ir.is_ok(), "{}", ir.unwrap_err());
}

#[test]
fn a_method_call_that_settles_on_a_binding_captures_it() {
    let i = Interner::new();
    let source = format!("{LEN_BINDING}range(0, 2) | map(|t| -> t.len()) | sum");
    let c = checked(&i, &source);
    assert_eq!(c.ret, Ty::Int(acvus_mir::ty::IntTy::I64));
    assert_eq!(c.captures_by_lambda.len(), 1, "{:?}", c.captures_by_lambda);
    assert!(
        matches!(c.captures_by_lambda[0].as_slice(), [Ty::Fn { .. }]),
        "{:?}",
        c.captures_by_lambda
    );
    let ir = compile_script_mode_optimized(&i, &source, &FxHashMap::default());
    assert!(ir.is_ok(), "{}", ir.unwrap_err());
}

#[test]
fn a_binding_read_beside_a_qualified_name_of_its_own_spelling_is_captured_once() {
    let i = Interner::new();
    let source = format!(
        "{LEN_BINDING}{LENT_ARRAY}as_iter(&a) | map(|k| -> array::len(k) as f64 + len(1) as f64) | sum"
    );
    let c = checked(&i, &source);
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(c.captures_by_lambda.len(), 1, "{:?}", c.captures_by_lambda);
    assert!(
        matches!(c.captures_by_lambda[0].as_slice(), [Ty::Fn { .. }]),
        "{:?}",
        c.captures_by_lambda
    );
    let ir = compile_script_mode_optimized(&i, &source, &FxHashMap::default());
    assert!(ir.is_ok(), "{}", ir.unwrap_err());
}

#[test]
fn a_star_on_a_captured_word_names_the_word_it_is_written_on() {
    let i = Interner::new();
    let errors = check(&i, "let k = 1.0; let f = |y| -> *k * y; f(2.0)")
        .expect_err("a captured word is not a reference");
    assert!(
        errors
            .iter()
            .any(|e| e == "`*` needs a reference, got Float"),
        "{errors:?}"
    );
}

/// `len` takes a `&String`, so a body that passes the captured name to it
/// with no `&` is reading it as the reference it is.
#[test]
fn a_captured_large_is_still_read_through_its_reference() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let s = \"a\".to_string(); let f = |u| -> len(s) + u; f(1)",
    );
    assert_eq!(c.ret, Ty::Int(acvus_mir::ty::IntTy::U64));
    assert!(
        c.captures_by_lambda
            .iter()
            .all(|captures| captures.as_slice() == [Ty::String]),
        "the closure owns the `String`: {:?}",
        c.captures_by_lambda
    );
}
