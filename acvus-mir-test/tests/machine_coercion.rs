//! There is deliberately no way for a script to reach the declarations the
//! compiler resolves for itself, and no way for a declaration to become one
//! by looking like one. `a[i]`, `for x in &v` and a `&str` parameter are
//! lowered through a view, and which declarations are views is a property
//! `#[extern_view]` puts on them and `TypeRegistry::machine_view` reads back.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    Effect, EffectTerm, Instances, Mutability, ParamTerm, Poly, PolyTy, Ty, TypeArg, lift_to_poly,
};
use acvus_mir_test::compile_script_mode_ir_with;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn shaped_like_the_str_view(i: &Interner, ns: &str, name: &str) -> Function {
    let ty = PolyTy::Fn {
        params: vec![ParamTerm::new(
            i.intern("s"),
            lift_to_poly(&Ty::Ref(
                Mutability::Shared,
                Box::new(TypeArg::uniform(Ty::String)),
            )),
        )],
        ret: Box::new(lift_to_poly(&Ty::Ref(
            Mutability::Shared,
            Box::new(TypeArg::uniform(Ty::Str)),
        ))),
        captures: vec![],
        effect: EffectTerm::<Poly>::Known(Effect::PURE),
    };
    Function {
        qref: QualifiedRef::qualified(i.intern(ns), i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
        },
        ty,
    }
}

fn compile(source: &str, declared_as: &str) -> Result<String, String> {
    let i = Interner::new();
    let declared = shaped_like_the_str_view(&i, "mine", declared_as);
    compile_script_mode_ir_with(&i, source, &FxHashMap::default(), &[declared])
}

fn calling(name: &str) -> String {
    format!("let s = \"hi\".to_string(); let v = mine::{name}(&s); 1")
}

#[test]
fn the_machines_spelling_does_not_make_a_declaration_the_machines() {
    let own_name = compile(&calling("view1"), "view1").expect("mine::view1 is a function");
    let machines_name = compile(&calling("as_str"), "as_str").expect("mine::as_str is a function");
    assert!(own_name.contains("view1"), "{own_name}");
    assert!(machines_name.contains("as_str"), "{machines_name}");
}

/// The one variable: the two views, offered back to the call by the name
/// the script wrote. `as_slice` has always been; `as_str` is now, as Rust
/// has both.
#[test]
fn both_machine_views_are_offered_back_at_the_call() {
    compile(
        "let v = vec([1, 2, 3]); let r = vec::as_slice(&v); 1",
        "view1",
    )
    .expect("vec::as_slice is offered back");
    compile(
        "let s = \"hi\".to_string(); let r = core::as_str(&s); 1",
        "view1",
    )
    .expect("core::as_str is offered back");
}
