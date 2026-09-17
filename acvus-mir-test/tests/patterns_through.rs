//! Intent tests for RFC-0024.

use acvus_extern::{Externs, TypesOnly};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{LenTerm, Mutability, PolyBuilder, Ty, TyTerm, TypeArg};
use acvus_mir_test::*;
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

fn items(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("items"),
        Ty::Array(Box::new(Ty::I64), LenTerm::Known(3)),
    )])
}

fn user(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("user"),
        Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::I64),
        ])),
    )])
}

#[test]
fn a_list_pattern_against_a_reference_binds_element_references() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "[a, b, ..] = &@items { x = *a + *b; }; 0", &items(&i)).unwrap();
    assert!(ir.contains("ref &(*r"), "{ir}");
    assert!(ir.contains("[0]"), "{ir}");
    assert!(ir.contains("commit @items"), "{ir}");
}

#[test]
fn a_word_binding_is_read_through_the_reference_at_an_operator() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "[a, b, ..] = &@items { x = a + 1; }; 0", &items(&i)).unwrap();
    assert!(ir.contains("take (*"), "{ir}");
}

#[test]
fn an_object_pattern_against_a_reference_binds_field_references() {
    let i = Interner::new();
    let ir = compile_to_ir(&i, "{{ { name, } = &@user }}{{ name }}{{/}}", &user(&i)).unwrap();
    assert!(ir.contains(".name"), "{ir}");
    assert!(ir.contains("string_concat"), "{ir}");
    assert!(ir.contains("commit @user"), "{ir}");
}

#[test]
fn a_literal_pattern_compares_through_the_reference() {
    let i = Interner::new();
    let role = FxHashMap::from_iter([(i.intern("role"), Ty::String)]);
    let ir = compile_to_ir(&i, r#"{{ "admin" = &@role }}yes{{_}}no{{/}}"#, &role).unwrap();
    assert!(ir.contains("test_literal") || ir.contains("test "), "{ir}");
    assert!(ir.contains("commit @role"), "{ir}");
}

#[test]
fn a_list_pattern_against_a_value_copies_its_words_out() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "[a, b, ..] = @items { x = a + b; }; 0", &items(&i)).unwrap();
    assert!(ir.contains("commit @items"), "{ir}");
}

#[test]
fn a_list_pattern_against_a_value_of_objects_leaves_it_partly_moved() {
    let i = Interner::new();
    let user = Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::I64)]));
    let users = FxHashMap::from_iter([(
        i.intern("users"),
        Ty::Array(Box::new(user), LenTerm::Known(2)),
    )]);
    let err = compile_script_ir(&i, "[a, b] = @users { x = 1; }; 0", &users).unwrap_err();
    assert!(
        err.contains("context @users is moved out here and not assigned again before the run ends"),
        "{err}"
    );
    let ir = compile_script_ir(&i, "[a, b] = &@users { x = a.age; }; 0", &users).unwrap();
    assert!(ir.contains("commit @users"), "{ir}");
}

#[test]
fn a_context_bind_through_a_reference_is_rejected() {
    let i = Interner::new();
    let ctx = FxHashMap::from_iter([
        (
            i.intern("user"),
            Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::I64)])),
        ),
        (
            i.intern("copy"),
            Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::I64)])),
        ),
    ]);
    let err = compile_script_ir(&i, "@copy = &@user; 0", &ctx).unwrap_err();
    assert!(!err.is_empty(), "{err}");
}

// -- An open head: the pattern settles when the scrutinee's head does ---
//
// A lambda's parameter has no type of its own at the pattern: the call
// gives it one, and the pattern is what the call's argument then decides
// (RFC-0024). `if let` is a script-mode expression and a lambda's body is
// an expression, so a pattern inside a lambda is written as the tag form
// `pattern = source { body };`.

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

/// Every type the checker recorded for the script, or its errors.
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
    let resolution = inf.outcomes[&qref].resolution().expect("complete");
    Ok(resolution.type_map.values().cloned().collect())
}

/// The types of the parameters of every lambda the script wrote.
fn lambda_parameter_types(types: &[Ty]) -> Vec<Ty> {
    let mut params: Vec<Ty> = types
        .iter()
        .filter_map(|ty| match ty {
            Ty::Fn { params, .. } => Some(params.iter().map(|p| p.ty.clone())),
            _ => None,
        })
        .flatten()
        .collect();
    params.sort_by_key(|ty| format!("{ty:?}"));
    params.dedup();
    params
}

fn shared_ref(ty: Ty) -> Ty {
    Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(ty)))
}

const LITERAL_ON_A_PARAMETER: &str = "f = |r| -> { out = 20; 1 = r { out = 10; }; out }; ";
const VARIANT_ON_A_PARAMETER: &str = "f = |r| -> { out = 0.0; Some(v) = r { out = *v; }; out }; ";
const VARIANT_READ_AS_A_VALUE: &str = "f = |r| -> { out = 0.0; Some(v) = r { out = v; }; out }; ";

#[test]
fn a_literal_against_an_open_head_compares_through_the_reference_the_call_lends() {
    let i = Interner::new();
    let source = format!("{LITERAL_ON_A_PARAMETER}a = 1; f(&a)");
    let types = recorded_types(&i, &source).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    assert_eq!(
        lambda_parameter_types(&types),
        vec![shared_ref(Ty::I64)],
        "the call lends `a`, so the pattern reads its scrutinee through the reference"
    );
    let ir = compile_script_ir(&i, &source, &FxHashMap::default()).unwrap();
    assert!(ir.contains("(r) : &i64"), "{ir}");
    assert!(
        ir.contains("== 1"),
        "the literal is compared, not moved out: {ir}"
    );
}

#[test]
fn a_variant_against_an_open_head_binds_a_reference_when_the_call_lends() {
    let i = Interner::new();
    let source = format!("{VARIANT_ON_A_PARAMETER}o = Some(1.5); f(&o)");
    let types = recorded_types(&i, &source).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    assert_eq!(
        lambda_parameter_types(&types),
        vec![shared_ref(Ty::Option(Box::new(Ty::Float)))],
        "the call lends `o`"
    );
    let ir = compile_script_ir(&i, &source, &FxHashMap::default()).unwrap();
    assert!(ir.contains("(v) : &Float"), "{ir}");
    assert!(
        ir.contains("ref &(*"),
        "the payload is named in place, not moved out: {ir}"
    );
}

#[test]
fn the_same_variant_against_a_value_binds_the_value() {
    let i = Interner::new();
    let source = format!("{VARIANT_READ_AS_A_VALUE}f(Some(1.5))");
    let types = recorded_types(&i, &source).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    assert_eq!(
        lambda_parameter_types(&types),
        vec![Ty::Option(Box::new(Ty::Float))],
        "the call passes the option itself"
    );
    assert!(
        !types.iter().any(|ty| matches!(ty, Ty::Ref(..))),
        "nothing in this script is a reference: {types:?}"
    );
}

#[test]
fn a_scrutinee_no_call_ever_fixes_closes_on_the_value() {
    let i = Interner::new();
    let source = format!("{VARIANT_READ_AS_A_VALUE}0");
    let types = recorded_types(&i, &source).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    assert!(
        !lambda_parameter_types(&types)
            .iter()
            .any(|ty| matches!(ty, Ty::Ref(..))),
        "a scrutinee nothing made a reference is not one: {types:?}"
    );
}

#[test]
fn a_binding_read_through_a_reference_before_the_mode_settles_refuses_the_value() {
    let i = Interner::new();
    let source = format!("{VARIANT_ON_A_PARAMETER}f(Some(1.5))");
    let errors = recorded_types(&i, &source).expect_err("`*v` asked for a reference `v` is not");
    assert!(
        errors.iter().any(|e| e.contains("type mismatch")),
        "{errors:?}"
    );
}

#[test]
fn a_variant_against_a_borrowed_option_binds_a_reference_and_leaves_it_owned() {
    let i = Interner::new();
    let source = "o = Some([1.0, 2.0]); out = 0.0; Some(v) = &o { out = *get(v, 1); }; out";
    let ir = compile_script_ir(&i, source, &FxHashMap::default()).unwrap();
    assert!(ir.contains("(v) : &Array<Float, 2>"), "{ir}");
    assert!(
        ir.contains("(o) : Option<Array<Float, 2>>"),
        "`o` is still the option it was: {ir}"
    );
}

/// A dump's instruction listing, without the value table that follows it.
fn instructions(ir: &str) -> String {
    ir.lines()
        .take_while(|line| !line.trim_start().starts_with(';'))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_tag_form_and_if_let_with_no_else_are_one_lowering() {
    let i = Interner::new();
    let out = FxHashMap::from_iter([(i.intern("out"), Ty::Float)]);
    let tag =
        compile_script_optimized(&i, "o = Some(1.5); Some(v) = o { @out = v; }; 0", &out).unwrap();
    let if_let = compile_script_mode_optimized(
        &i,
        "let o = Some(1.5); if let Some(v) = o { @out = v; }; 0",
        &out,
    )
    .unwrap();
    assert_eq!(
        instructions(&tag),
        instructions(&if_let),
        "both go through `lower_match_bind_arm`"
    );
    // The one difference is outside the listing: `if let` is an expression
    // and its value is the Unit no instruction reads. The tag form is a
    // statement and has none.
    assert!(if_let.contains(": Unit"), "{if_let}");
    assert!(!tag.contains(": Unit"), "{tag}");
}

#[test]
fn a_context_bound_by_a_pattern_whose_head_settles_on_a_reference_is_refused() {
    let i = Interner::new();
    let ctx = FxHashMap::from_iter([
        (
            i.intern("user"),
            Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::I64)])),
        ),
        (
            i.intern("copy"),
            Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::I64)])),
        ),
    ]);
    let err = compile_script_ir(
        &i,
        "f = |r| -> { @copy = r { x = 1; }; 0 }; f(&@user)",
        &ctx,
    )
    .unwrap_err();
    assert!(
        err.contains("a reference cannot be stored in a list, object, or tuple"),
        "a context holds data (RFC-0014): {err}"
    );
}
