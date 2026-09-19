//! A `&str` parameter takes `&s` by coercion (RFC-0062 Decision 3): where
//! the `String`'s `as_str` ends up, and what a referent that is no `String`
//! is refused with.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    Effect, EffectTerm, Instances, Mutability, ParamTerm, Poly, PolyTy, Ty, TypeArg, lift_to_poly,
};
use acvus_mir_test::compile_script_mode_ir_with;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn str_view() -> Ty {
    Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::Str)))
}

fn takes_str(i: &Interner, name: &str) -> Function {
    let ty = PolyTy::Fn {
        params: vec![ParamTerm::new(i.intern("s"), lift_to_poly(&str_view()))],
        ret: Box::new(lift_to_poly(&Ty::U64)),
        captures: vec![],
        effect: EffectTerm::<Poly>::Known(Effect::PURE),
    };
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
        },
        ty,
    }
}

fn ir(source: &str) -> String {
    let i = Interner::new();
    let externs = [takes_str(&i, "byte_length")];
    compile_script_mode_ir_with(&i, source, &FxHashMap::default(), &externs)
        .unwrap_or_else(|e| panic!("{source}\n{e}"))
}

fn refusal(source: &str) -> String {
    let i = Interner::new();
    let externs = [takes_str(&i, "byte_length")];
    let reported = compile_script_mode_ir_with(&i, source, &FxHashMap::default(), &externs)
        .expect_err(&format!("{source} is refused"));
    reported
        .lines()
        .map(|line| line.split("] ").nth(1).unwrap_or(line))
        .collect::<Vec<&str>>()
        .join("; ")
}

fn main_body(ir: &str) -> &str {
    ir.split("=== main ===")
        .nth(1)
        .expect("an entry body")
        .split("=== ")
        .next()
        .expect("the entry body ends at the next section")
}

#[test]
fn a_borrowed_string_at_a_str_parameter_is_one_view_before_the_call() {
    let ir = ir("let s = \"abc\".to_string(); byte_length(&s)");
    let body = main_body(&ir);
    assert_eq!(body.matches("as_slice").count(), 1, "{body}");
    let viewed = body
        .lines()
        .position(|line| line.contains("as_slice"))
        .expect("the view");
    let called = body
        .lines()
        .position(|line| line.contains("byte_length"))
        .expect("the call");
    assert!(viewed < called, "the view is the argument:\n{body}");
}

#[test]
fn a_referent_that_is_no_string_is_the_argument_mismatch_it_is() {
    let reported = refusal("let n = 1; byte_length(&n)");
    assert!(
        reported.contains("str"),
        "the refusal names the parameter: {reported}"
    );
}
