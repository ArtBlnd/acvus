//! Qualified calls and method calls (RFC-0030): `ns::f(args)` names a
//! namespace when one declares `f`, `recv.f(args)` is `f(recv', args)`
//! with the receiver lent as `f`'s first parameter asks.

use acvus_mir::ty::Ty;
use acvus_mir_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn check(src: &str) -> Result<String, String> {
    let i = Interner::new();
    compile_script_optimized(&i, src, &FxHashMap::default())
}

fn tail_ty(src: &str) -> Ty {
    let i = Interner::new();
    let ir =
        compile_script_optimized(&i, src, &FxHashMap::default()).unwrap_or_else(|e| panic!("{e}"));
    let line = ir
        .lines()
        .find(|l| l.contains("return "))
        .expect("a return of the entry body, which is printed before any closure's");
    let reg = line.split("return ").nth(1).unwrap().trim();
    let ty = ir
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("; {reg} (")))
        .expect("the returned value's type");
    match ty.split(" : ").nth(1).unwrap().trim() {
        "i64" => Ty::I64,
        "u64" => Ty::Int(acvus_mir::ty::IntTy::U64),
        "String" => Ty::String,
        other => panic!("unexpected tail type {other}"),
    }
}

#[test]
fn a_qualified_name_is_a_call_when_its_namespace_declares_the_function() {
    let ir = check("let xs = [1, 2]; array::len(&xs)").expect("array::len");
    assert!(ir.contains("call "), "{ir}");
    assert!(!ir.contains("variant"), "{ir}");
    check("let d = deque(); deque::push_back(&mut d, 1); deque::push_back(&mut d, 2); len(&d)")
        .expect("a qualified call takes any number of arguments");
}

#[test]
fn a_qualified_name_no_namespace_declares_is_a_structural_variant() {
    let ir = check("Shape::Circle(1)").expect("the variant it was");
    assert!(ir.contains("variant Circle"), "{ir}");
    let err = check("Shape::Circle(1, 2)").expect_err("a variant has one payload");
    assert!(err.contains("Shape::Circle"), "{err}");
    let ir = check("let xs = [1]; nope::len(&xs)").expect("`nope` is no namespace: a variant");
    assert!(ir.contains("variant len"), "{ir}");
}

#[test]
fn a_method_call_lends_its_receiver_as_the_callee_s_first_parameter_asks() {
    assert_eq!(
        tail_ty("let xs = [1, 2, 3]; xs.len()"),
        Ty::Int(acvus_mir::ty::IntTy::U64)
    );
    let ir = check("let xs = [1, 2, 3]; xs.len()").unwrap();
    assert!(ir.contains("ref &xs"), "{ir}");
    let ir = check("let d = deque(); d.push_back(1); d.len()").unwrap();
    assert!(ir.contains("ref &mut d"), "{ir}");
    assert_eq!(
        tail_ty("let xs = [1, 2, 3]; let ys = xs.as_iter().map(|x| -> *x * 2).collect(); ys.len()"),
        Ty::Int(acvus_mir::ty::IntTy::U64)
    );
    let ir = check("let d = deque(); d.push_back({ x: 1, }); d.get(0).x").unwrap();
    assert!(ir.contains("ref &d"), "{ir}");
}

#[test]
fn a_method_receiver_that_must_be_lent_is_a_place() {
    let err = check("[1, 2].len()").expect_err("a temporary is not a place");
    assert!(err.contains("can be referenced"), "{err}");
}

#[test]
fn a_method_call_of_a_signature_picks_the_instance_by_the_receiver() {
    let ir = check("let xs = [1, 2, 3]; xs.into_iter().fold(0, |a, x| -> a + x)").unwrap();
    assert!(ir.contains("call "), "{ir}");
    assert_eq!(tail_ty("let s = \"ab\".to_string(); s.clone()"), Ty::String);
}

#[test]
fn a_receiver_that_is_already_a_reference_is_passed_as_it_is() {
    let ir = check("let xs = [\"a\".to_string(), \"b\".to_string()]; xs[1].clone()")
        .expect("`xs[1]` gives `&String`, `clone` takes `&T`");
    assert_eq!(
        ir.matches("ref &").count(),
        1,
        "xs is lent for its slice; the element is already a `&String`, so it is \
         the receiver and no reborrow of it is built:\n{ir}"
    );
    assert_eq!(
        tail_ty("let xs = [\"a\".to_string(), \"b\".to_string()]; xs[1].clone()"),
        Ty::String
    );
}
