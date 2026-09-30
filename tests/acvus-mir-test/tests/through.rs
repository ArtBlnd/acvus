//! Intent tests for RefTarget::Through (RFC-0018).

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    Mutability, ObjectTy, Param, ParamTerm, Poly, Ty, TyTerm, TypeArg, lift_to_poly,
};
use acvus_mir_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn extern_fn(i: &Interner, name: &str, params: &[Ty], ret: Ty) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
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
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }
}

fn user(i: &Interner) -> Ty {
    Ty::Object(ObjectTy::written(FxHashMap::from_iter([
        (i.intern("name"), Ty::String),
        (i.intern("age"), Ty::I64),
    ])))
}

fn closure_taking(i: &Interner, param: Ty, ret: Ty) -> Ty {
    Ty::Fn {
        params: vec![Param::new(i.intern("u"), param)],
        ret: Box::new(ret),
        captures: vec![],
        effect: acvus_mir::ty::Effect::OPAQUE.into(),
        flows: acvus_mir::ty::Flows::Every.into(),
    }
}

fn with_closure(i: &Interner, param: Ty, ret: Ty, body: &str) -> Result<String, String> {
    let f = extern_fn(&i, "f", &[closure_taking(i, param, ret)], Ty::I64);
    compile_script_ir_with(i, &format!("f(|u| -> {body})"), &FxHashMap::default(), &[f])
}

#[test]
fn a_primitive_field_is_read_through_a_reference() {
    let i = Interner::new();
    let shared = Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(user(&i))));
    let ir = with_closure(&i, shared, Ty::I64, "u.age").unwrap();
    assert!(ir.contains("take (*r"), "{ir}");
}

#[test]
fn a_string_field_is_copied_through_a_reference() {
    let i = Interner::new();
    let shared = Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(user(&i))));
    let ir = with_closure(&i, shared, Ty::String, "u.name").unwrap();
    assert!(ir.contains("take (*r"), "{ir}");
}

#[test]
fn an_object_field_cannot_be_moved_out_through_a_reference() {
    let i = Interner::new();
    let nested = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
        i.intern("inner"),
        Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("age"),
            Ty::I64,
        )]))),
    )])));
    let shared = Ty::Ref(
        Mutability::Shared,
        Box::new(TypeArg::uniform(nested.clone())),
    );
    let inner = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
        i.intern("age"),
        Ty::I64,
    )])));
    let err = with_closure(&i, shared, inner, "u.inner").unwrap_err();
    assert!(!err.is_empty(), "{err}");
}

#[test]
fn a_string_field_is_borrowed_through_a_reference_and_cloned() {
    let i = Interner::new();
    let shared = Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(user(&i))));
    let ir = with_closure(&i, shared, Ty::Bool, r#"u.name == "bob""#).unwrap();
    assert!(ir.contains("ref &(*r"), "{ir}");
    assert!(ir.contains("string_eq"), "{ir}");
}

#[test]
fn a_field_is_assigned_through_a_mutable_reference() {
    let i = Interner::new();
    let mutable = Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(user(&i))));
    let ir = with_closure(&i, mutable, Ty::I64, "{ u.age = 1; 0 }").unwrap();
    assert!(ir.contains("assign (*r"), "{ir}");
}

#[test]
fn a_field_cannot_be_assigned_through_a_shared_reference() {
    let i = Interner::new();
    let shared = Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(user(&i))));
    let err = with_closure(&i, shared, Ty::I64, "{ u.age = 1; 0 }").unwrap_err();
    assert!(err.contains("not a `&mut`"), "{err}");
}

#[test]
fn a_word_is_stored_through_a_mutable_reference() {
    let i = Interner::new();
    let mutable = Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::I64)));
    let ir = with_closure(&i, mutable, Ty::I64, "{ *u = 1; *u }").unwrap();
    assert!(ir.contains("assign (*r"), "{ir}");
    assert!(ir.contains("take (*r"), "{ir}");
}

#[test]
fn a_word_cannot_be_stored_through_a_shared_reference() {
    let i = Interner::new();
    let shared = Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::I64)));
    let err = with_closure(&i, shared, Ty::I64, "{ *u = 1; 0 }").unwrap_err();
    assert!(err.contains("not a `&mut`"), "{err}");
}
