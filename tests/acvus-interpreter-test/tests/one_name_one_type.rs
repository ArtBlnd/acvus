//! One acvus type name is one Rust type, refused where the registries are
//! combined: a value's acvus type determines the box it is in.

use acvus_extern::{
    CombineError, ExternType, Externs, Registry, TyArg, extern_fn, extern_registry,
};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::*;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn combined(registries: Vec<Registry<AcvusRuntime>>) -> Result<(), CombineError> {
    let i = Interner::new();
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.extend(registries);
    Externs::combine(regs, &i).map(|_| ())
}

fn refusal(registries: Vec<Registry<AcvusRuntime>>) -> CombineError {
    combined(registries)
        .err()
        .expect("the registries give one name two Rust types")
}

// =======================================================================
//  Two extension types under one name
// =======================================================================

mod twin_a {
    use acvus_extern::ExternType;

    #[derive(ExternType)]
    #[extern_type(name = "Twin")]
    #[repr(transparent)]
    pub struct Twin(pub Vec<i64>);
}

mod twin_b {
    use acvus_extern::ExternType;

    #[derive(ExternType)]
    #[extern_type(name = "Twin")]
    #[repr(transparent)]
    pub struct Twin(pub String);
}

#[extern_fn(effect = pure)]
fn make_twin_a(n: i64) -> twin_a::Twin {
    twin_a::Twin(vec![n, n])
}

#[extern_fn(effect = pure)]
fn twin_b_len(t: twin_b::Twin) -> i64 {
    i64::try_from(t.0.len()).expect("a string shorter than i64::MAX")
}

fn twin_a_registry() -> Registry<AcvusRuntime> {
    extern_registry! { ns: "twin_a", types: [twin_a::Twin], fns: [make_twin_a] }
}

fn twin_b_registry() -> Registry<AcvusRuntime> {
    extern_registry! { ns: "twin_b", types: [twin_b::Twin], fns: [] }
}

fn twin_b_reader() -> Registry<AcvusRuntime> {
    extern_registry! { ns: "twin_b_reader", types: [], fns: [twin_b_len] }
}

fn duplicate_named(err: CombineError, wanted: &str) {
    match err {
        CombineError::DuplicateName { name } => assert_eq!(name, wanted),
        other => panic!("expected `{wanted}` declared twice, got {other}"),
    }
}

#[test]
fn two_registries_declaring_two_rust_types_under_one_name_are_refused() {
    duplicate_named(refusal(vec![twin_a_registry(), twin_b_registry()]), "Twin");
}

#[test]
fn a_declaration_reading_another_rust_type_under_a_declared_name_is_refused() {
    duplicate_named(refusal(vec![twin_a_registry(), twin_b_reader()]), "Twin");
}

#[test]
fn a_declaration_naming_a_type_no_registry_declares_is_refused() {
    match refusal(vec![twin_b_reader()]) {
        CombineError::UndeclaredType { declaration, ty } => {
            assert_eq!(declaration, "twin_b_reader::twin_b_len");
            assert_eq!(ty, "Twin");
        }
        other => panic!("expected an undeclared `Twin`, got {other}"),
    }
}

#[test]
fn one_rust_type_declared_by_two_registries_combines() {
    fn again() -> Registry<AcvusRuntime> {
        extern_registry! { ns: "twin_a_again", types: [twin_a::Twin], fns: [] }
    }
    combined(vec![twin_a_registry(), again()]).expect("one Rust type under one name");
}

/// `acvus_ext::std_registries` declares `Vec`, so this is its second
/// declaration.
#[test]
fn vec_declared_again_combines() {
    fn vec_again() -> Registry<AcvusRuntime> {
        extern_registry! { ns: "vec_again", types: [Vec<_>], fns: [] }
    }
    combined(vec![vec_again()]).expect("`Vec` is one Rust type");
}

// =======================================================================
//  Two derived structs under one name
// =======================================================================

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "PBag")]
#[extern_type(unsafe(uniform_payload))]
pub struct PBag<T>(pub Vec<T>)
where
    T: acvus_extern::Var<acvus_extern::kind::Type>;

mod rooted {
    use acvus_extern::TyArg;

    #[derive(TyArg, Debug)]
    pub struct P {
        pub x: i64,
    }
}

mod clashing {
    use acvus_extern::TyArg;

    #[derive(TyArg, Debug)]
    pub struct P {
        pub x: i64,
    }
}

mod spaced {
    use acvus_extern::TyArg;

    #[derive(TyArg, Debug)]
    #[ty_arg(ns = "spaced")]
    pub struct P {
        pub x: i64,
    }
}

#[extern_fn(effect = pure)]
fn rooted_bag(n: i64) -> PBag<rooted::P> {
    PBag(vec![rooted::P { x: n }])
}

#[extern_fn(effect = pure)]
fn rooted_bag_x(bag: PBag<rooted::P>) -> i64 {
    bag.0.first().expect("a bag of one").x
}

#[extern_fn(effect = pure)]
fn clashing_bag_x(bag: PBag<clashing::P>) -> i64 {
    bag.0.first().expect("a bag of one").x
}

#[extern_fn(effect = pure)]
fn spaced_bag(n: i64) -> PBag<spaced::P> {
    PBag(vec![spaced::P { x: n * 10 }])
}

#[extern_fn(effect = pure)]
fn spaced_bag_x(bag: PBag<spaced::P>) -> i64 {
    bag.0.first().expect("a bag of one").x
}

fn clashing_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "bags",
        types: [PBag<_>],
        fns: [rooted_bag, clashing_bag_x],
    }
}

fn spaced_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "bags",
        types: [PBag<_>],
        fns: [rooted_bag, rooted_bag_x, spaced_bag, spaced_bag_x],
    }
}

#[test]
fn two_derived_structs_of_one_name_reached_from_one_registry_are_refused() {
    duplicate_named(refusal(vec![clashing_registry()]), "P");
}

#[test]
fn a_derived_struct_under_a_namespace_is_its_own_name() {
    combined(vec![spaced_registry()]).expect("`P` and `spaced::P` are two names");
}

fn spaced_refusal_at(source: &str, opt: Opt) -> String {
    let i = Interner::new();
    let ast = acvus_mir::graph::ParsedAst::Script(
        acvus_ast::parse_script(&i, source).expect("parse error"),
    );
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(spaced_registry());
    match check_source(&i, ast, &FxHashMap::default(), regs, Ty::I64, opt, |_| {}) {
        Ok(_) => panic!("the program was admitted at {opt:?}: {source}"),
        Err(refusal) => refusal.messages.join(" | "),
    }
}

#[test]
fn a_bag_of_one_derived_struct_is_refused_where_the_other_is_read() {
    for opt in [Opt::None, Opt::Full] {
        spaced_refusal_at("spaced_bag_x(rooted_bag(7))", opt);
        spaced_refusal_at("rooted_bag_x(spaced_bag(7))", opt);
    }
}

async fn run_spaced(source: &str) -> i64 {
    let i = Interner::new();
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(spaced_registry());
    run_script_with_externs(&i, source, Context::default(), regs, Ty::I64)
        .await
        .value
        .as_int()
}

#[tokio::test]
async fn each_derived_struct_is_read_where_it_was_made() {
    assert_eq!(run_spaced("rooted_bag_x(rooted_bag(7))").await, 7);
    assert_eq!(run_spaced("spaced_bag_x(spaced_bag(7))").await, 70);
}

#[test]
fn a_derived_struct_names_its_namespace() {
    let i = Interner::new();
    let vars = acvus_extern::PolyVars::empty();
    let Ty::Object(object) =
        acvus_extern::try_freeze_poly(&<spaced::P as TyArg>::poly_ty(&i, &vars))
            .expect("a derived struct has no variable")
    else {
        panic!("a derived struct is an object type")
    };
    assert_eq!(
        object.declaration(),
        Some(acvus_utils::QualifiedRef::qualified(
            i.intern("spaced"),
            i.intern("P")
        ))
    );
}

#[test]
fn a_declaration_keeps_its_namespace_through_serialization() {
    let i = Interner::new();
    let spaced = acvus_utils::QualifiedRef::qualified(i.intern("spaced"), i.intern("P"));
    let object = Ty::Object(acvus_mir::ty::ObjectTy::declared(
        spaced,
        [(i.intern("x"), Ty::I64)].into_iter().collect(),
    ));
    let enumeration = Ty::Enum {
        name: spaced,
        variants: [(i.intern("A"), None)].into_iter().collect(),
        home: acvus_mir::ty::Home::NONE,
    };
    for ty in [object, enumeration] {
        let json = serde_json::to_string(&ty.to_ser(&i)).expect("serialize");
        let back: acvus_mir::ser_ty::SerTy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.to_ty(&i), ty, "{json}");
    }
}

#[test]
fn a_root_declaration_is_written_without_a_namespace() {
    let i = Interner::new();
    let object = Ty::Object(acvus_mir::ty::ObjectTy::declared(
        acvus_utils::QualifiedRef::root(i.intern("P")),
        [(i.intern("x"), Ty::I64)].into_iter().collect(),
    ));
    let json = serde_json::to_string(&object.to_ser(&i)).expect("serialize");
    assert!(json.contains(r#"{"kind":"declared","name":"P"}"#), "{json}");
}
