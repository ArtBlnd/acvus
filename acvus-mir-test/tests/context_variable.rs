//! Intent tests for RFC-0025.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{Mutability, Param, ParamTerm, Poly, Ty, TyTerm, TypeArg, lift_to_poly};
use acvus_mir_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn extern_fn(i: &Interner, name: &str, params: &[Ty], ret: Ty) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Default::default(),
        },
        ty: TyTerm::Fn {
            params: params
                .iter()
                .enumerate()
                .map(|(n, ty)| ParamTerm::<Poly>::new(i.intern(&format!("_{n}")), lift_to_poly(ty)))
                .collect(),
            ret: Box::new(lift_to_poly(&ret)),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    }
}

fn int_to_int(i: &Interner) -> Ty {
    Ty::Fn {
        params: vec![Param::new(i.intern("a"), Ty::I64)],
        ret: Box::new(Ty::I64),
        captures: vec![],
        effect: acvus_mir::ty::Effect::OPAQUE.into(),
    }
}

fn string_context(i: &Interner, name: &str) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(i.intern(name), Ty::String)])
}

#[test]
fn an_assign_to_a_context_while_it_is_lent_is_rejected() {
    let i = Interner::new();
    let f = extern_fn(
        &i,
        "f",
        &[
            Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::String))),
            Ty::I64,
        ],
        Ty::I64,
    );
    let err = compile_script_ir_with(
        &i,
        r#"f(&@items, { @items = "x"; 1 })"#,
        &string_context(&i, "items"),
        &[f],
    )
    .unwrap_err();
    assert!(err.contains("is touched while the reference"), "{err}");
}

#[test]
fn a_closure_writing_a_lent_context_is_rejected_at_the_call() {
    let i = Interner::new();
    let f = extern_fn(
        &i,
        "f",
        &[
            Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::String))),
            int_to_int(&i),
        ],
        Ty::I64,
    );
    let err = compile_script_ir_with(
        &i,
        r#"f(&@items, |x| -> { @items = "x"; x })"#,
        &string_context(&i, "items"),
        &[f],
    )
    .unwrap_err();
    assert!(err.contains("is touched while the reference"), "{err}");
}

#[test]
fn a_context_read_after_a_call_whose_closure_writes_it_is_fetched_again() {
    let i = Interner::new();
    let f = extern_fn(&i, "f", &[int_to_int(&i)], Ty::I64);
    let ctx = FxHashMap::from_iter([(i.intern("x"), Ty::I64)]);
    let ir = compile_script_ir_with(&i, "@x = 1; f(|a| -> { @x = 2; a }); @x", &ctx, &[f]).unwrap();
    let main = ir.split("=== closure").next().unwrap();
    let commit = main
        .find("commit @x")
        .expect("the context is committed before the call");
    let call = main.find("call #").expect("the call");
    let fetch = main
        .rfind("fetch @x")
        .expect("the context is fetched after the call");
    assert!(commit < call && call < fetch, "{main}");
    let returned = main.lines().find(|l| l.contains("return")).unwrap();
    assert!(!returned.contains("return 1"), "{main}");
}

fn object_context(i: &Interner, name: &str) -> FxHashMap<acvus_utils::Astr, Ty> {
    let user = Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::I64)]));
    FxHashMap::from_iter([(i.intern(name), user)])
}

#[test]
fn a_context_moved_out_and_not_assigned_back_is_rejected() {
    let i = Interner::new();
    let err = compile_script_ir(&i, "let x = @user; x", &object_context(&i, "user")).unwrap_err();
    assert!(
        err.contains("context @user is moved out here and not assigned again before the run ends"),
        "{err}"
    );
}

#[test]
fn a_context_bound_and_read_again_is_rejected_before_promotion() {
    let i = Interner::new();
    let err = compile_script_ir(
        &i,
        "let x = @user; let y = @user; 0",
        &object_context(&i, "user"),
    )
    .unwrap_err();
    assert!(err.contains("after it was moved"), "{err}");
}

#[test]
fn a_context_moved_out_and_assigned_back_is_accepted() {
    let i = Interner::new();
    compile_script_ir(
        &i,
        "let x = @user; @user = { age: 1, }; x",
        &object_context(&i, "user"),
    )
    .unwrap();
}

#[test]
fn a_string_context_named_twice_is_copied_before_its_first_use() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "let x = @items; x", &string_context(&i, "items")).unwrap();
    assert!(ir.contains("string_clone"), "{ir}");
}

#[test]
fn a_string_context_used_once_after_reassignment_is_not_copied() {
    let i = Interner::new();
    let ir = compile_script_ir(
        &i,
        r#"let x = @items; @items = "new"; x"#,
        &string_context(&i, "items"),
    )
    .unwrap();
    assert!(!ir.contains("string_clone"), "{ir}");
}

#[test]
fn a_template_that_binds_a_string_context_and_emits_it_is_accepted() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items }}{{ x }}"#,
        &string_context(&i, "items"),
    )
    .unwrap();
    assert!(ir.contains("string_clone"), "{ir}");
}
