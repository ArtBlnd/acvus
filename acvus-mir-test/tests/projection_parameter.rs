//! A projection parameter is matched at least (RFC-0050 rule 6): a handler
//! that declares `SRef<'_>` borrows the fields the projection names, so an
//! object carrying more of them is admitted. A by-value `S` parameter keeps
//! RFC-0042 R1's exact meet, which `declared_struct.rs` is the contract of.
//!
//! What `#[derive(TyArg)] #[projection]` emits is what these externs declare:
//! the struct's own `TyArg` is `ObjectTy::declared`, and the projection's is
//! `&` over `ObjectTy::at_least` of the same fields.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{Effect, Instances, Mutability, ObjectTy, ParamTerm, Poly, TyTerm, TypeArg};
use acvus_mir_test::compile_script_mode_ir_with;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn fields(i: &Interner, names: &[&str]) -> FxHashMap<Astr, TyTerm<Poly>> {
    names
        .iter()
        .map(|name| (i.intern(name), TyTerm::I64))
        .collect()
}

fn point_ty(i: &Interner) -> TyTerm<Poly> {
    TyTerm::Object(ObjectTy::declared(
        i.intern("Point"),
        fields(i, &["x", "y"]),
    ))
}

fn at_least_ref(i: &Interner, names: &[&str]) -> TyTerm<Poly> {
    let at_least = TyTerm::Object(ObjectTy::at_least(fields(i, names)));
    TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(at_least)))
}

fn extern_fn(i: &Interner, name: &str, param: Option<TyTerm<Poly>>, ret: TyTerm<Poly>) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
        },
        ty: TyTerm::Fn {
            params: param
                .into_iter()
                .map(|ty| ParamTerm::<Poly>::new(i.intern("p"), ty))
                .collect(),
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::PURE.into(),
        },
    }
}

fn compile(i: &Interner, source: &str) -> Result<String, String> {
    let externs = [
        extern_fn(i, "take_point", Some(point_ty(i)), TyTerm::I64),
        extern_fn(
            i,
            "read_point",
            Some(at_least_ref(i, &["x", "y"])),
            TyTerm::I64,
        ),
        extern_fn(i, "read_y", Some(at_least_ref(i, &["y"])), TyTerm::I64),
        extern_fn(i, "read_z", Some(at_least_ref(i, &["z"])), TyTerm::I64),
        extern_fn(i, "point", None, point_ty(i)),
    ];
    compile_script_mode_ir_with(i, source, &FxHashMap::default(), &externs)
}

fn refusal(i: &Interner, source: &str) -> String {
    compile(i, source).expect_err("the projection fixes what the object must have")
}

#[test]
fn a_projection_admits_an_object_of_exactly_the_fields_it_names() {
    let i = Interner::new();
    compile(&i, "let p = { x: 1, y: 2, }; read_point(&p)")
        .expect("every field the projection names");
}

#[test]
fn a_projection_admits_an_object_carrying_a_field_it_does_not_name() {
    let i = Interner::new();
    compile(&i, "let p = { x: 1, y: 2, extra: 3, }; read_point(&p)")
        .expect("a projection borrows the fields it names and no others");
}

#[test]
fn a_partial_projection_admits_an_object_of_more_fields() {
    let i = Interner::new();
    compile(&i, "let p = { x: 1, y: 2, }; read_y(&p)").expect("one field of a two-field object");
}

#[test]
fn a_projection_admits_a_value_of_a_declared_type_that_has_its_fields() {
    let i = Interner::new();
    compile(&i, "let p = point(); read_y(&p)").expect("`Point` has `y`");
}

#[test]
fn a_projection_naming_a_field_a_declared_type_lacks_is_refused_by_that_name() {
    let i = Interner::new();
    let err = refusal(&i, "let p = point(); read_z(&p)");
    assert!(
        err.contains("object has field `z` that `Point` does not declare"),
        "{err}"
    );
}

/// The gap RFC-0050 rule 6 has at a projection parameter, stated as the
/// checker's actual answer. Rule 6 says a projection naming a field the
/// object lacks is refused. That holds where the argument's type is
/// `Declared`, which the test above pins, and **not** where it is an object
/// literal: the literal is admitted and its type does not even grow the
/// field, so `p` stays `{x: i64}`. The glue then asks the object's shape for
/// `y` at call time and panics, which `projection::position_of` is the site
/// of. Closing it is a change to how `ObjectTy::meet` joins `Written` with
/// `AtLeast`, and that join is RFC-0042's own rule that a field store adds
/// to a literal's field set, so it is not this test's to decide.
#[test]
fn an_object_literal_lacking_a_projections_field_is_admitted_unchanged() {
    let i = Interner::new();
    let ir = compile(&i, "let p = { x: 1, }; read_y(&p)").expect("the literal is admitted");
    assert!(ir.contains("r2 (p.) : &{x: i64}"), "{ir}");
}

/// The by-value parameter is the exact meet, and the object the projection
/// admits above is refused here by the field it adds.
#[test]
fn a_by_value_parameter_keeps_the_exact_meet() {
    let i = Interner::new();
    compile(&i, "take_point({ x: 1, y: 2, })").expect("the declared field set");
    let err = refusal(&i, "take_point({ x: 1, y: 2, extra: 3, })");
    assert!(
        err.contains("object has field `extra` that `Point` does not declare"),
        "{err}"
    );
}
