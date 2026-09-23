//! A projection parameter is matched at least (RFC-0050 rule 6): a handler
//! that declares `SRef<'_>` borrows the fields the projection names, so an
//! object carrying more of them is admitted. A by-value `S` parameter keeps
//! RFC-0042 rule 1's exact meet, which `declared_struct.rs` is the contract of.
//!
//! What `#[derive(TyArg)] #[projection]` emits is what these externs declare:
//! the struct's own `TyArg` is `ObjectTy::declared`, and the projection's is
//! `&` over `ObjectTy::at_least` of the same fields.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{Effect, Instances, Mutability, ObjectTy, ParamTerm, Poly, TyTerm, TypeArg};
use acvus_mir_test::{Marked, Refusal, compile_script_mode_ir_with, refuse_script_mode_ir_with};
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
        QualifiedRef::root(i.intern("Point")),
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
            effect_bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
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

fn externs(i: &Interner) -> [Function; 5] {
    [
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
    ]
}

fn compile(i: &Interner, source: &str) -> Result<String, String> {
    compile_script_mode_ir_with(i, source, &FxHashMap::default(), &externs(i))
}

fn refusal(i: &Interner, source: &str) -> String {
    compile(i, source).expect_err("the projection fixes what the object must have")
}

/// The refusals of `source`, each with the labels it points at.
fn refusals(i: &Interner, source: &str) -> Vec<Refusal> {
    let externs = externs(i);
    match refuse_script_mode_ir_with(i, source, &FxHashMap::default(), &externs) {
        Ok(ir) => panic!("expected a refusal, compiled:\n{ir}"),
        Err(refusals) => refusals,
    }
}

fn words(refusals: &[Refusal]) -> Vec<String> {
    refusals.iter().map(Refusal::to_string).collect()
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

/// A literal's field set is exactly what it wrote, so a projection naming a
/// field it lacks is refused as a `Declared` value lacking one already is.
/// The refusal marks the call and labels the argument: the registry holds no
/// declaration span for an extern parameter, so the call is the only place
/// the callee's side can be pointed at.
/// The union `ObjectTy::meet` gives a `Written` set stays where RFC-0042
/// means it -- a field store widens a literal -- and a parameter stores
/// nothing.
#[test]
fn a_projection_naming_a_field_an_object_literal_lacks_is_refused() {
    let i = Interner::new();
    let source = "let p = { x: 1, }; read_y(&p)";
    let refusals = refusals(&i, source);
    assert_eq!(refusals.len(), 1, "{:#?}", words(&refusals));
    assert_eq!(
        refusals[0].message,
        "`p` lacks field `y`, which the projection parameter borrows"
    );
    assert_eq!(refusals[0].at(source), "read_y(&p)");
    assert_eq!(
        refusals[0].marked(source),
        [
            Marked {
                source: Some("&p".to_string()),
                text: "this argument".to_string(),
            },
            Marked {
                source: None,
                text: "the parameter borrows at least `{y: i64}`, and `p` has `{x: i64}`"
                    .to_string(),
            }
        ]
    );
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
