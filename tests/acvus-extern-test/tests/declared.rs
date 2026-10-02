//! `Declared`: a Rust type's acvus type as a declaration states it, the Rust
//! type as written. `Uniform<R, X>` declares `X`, a `Var<N>` stand-in its
//! variable, and a whole extern-defined type written natively is marked `#`.

use acvus_extern::{EffectAt, EffectVar, IdentityAt, IdentityVar, Opaque, Structural, Uniform, Var};
use acvus_extern_test::Safe;
use acvus_mir::ty::{EffectTerm, IdentityTerm, IntTy, Mark, MarkTerm, Mutability, TyKind};
use acvus_utils::{Interner, QualifiedRef};

type U<X> = Uniform<Safe, X>;

fn runtime() -> Safe {
    Safe::new(Interner::new())
}

#[derive(Structural)]
struct Point {
    b: String,
    a: i64,
}

#[derive(Structural)]
#[allow(dead_code)]
enum Shape {
    Square(f64),
    Empty,
    Rect(f64, f64),
    Labeled { width: f64, name: String },
}

#[derive(Opaque)]
#[allow(dead_code)]
struct Stack<T> {
    items: Vec<T>,
}

#[derive(Opaque)]
#[allow(dead_code)]
struct Counter {
    n: i64,
}

#[derive(Opaque)]
#[allow(dead_code)]
struct Cursor<'a, T> {
    items: &'a [T],
}

#[derive(Opaque)]
#[allow(dead_code)]
struct Session<E: EffectVar, I>
where
    I: IdentityVar,
{
    effect: std::marker::PhantomData<(E, I)>,
}

#[test]
fn leaves_and_structures_declare_as_written() {
    let mut rt = runtime();
    let int = rt.declare::<i64>();
    assert_eq!(rt.types().kind(int), &TyKind::Int(IntTy::I64));
    let text = rt.declare::<&str>();
    let str = rt.declare::<str>();
    assert_eq!(rt.types().kind(text), &TyKind::Ref(Mutability::Shared, str));
    let run = rt.declare::<&mut [i64]>();
    let slice = rt.declare::<[i64]>();
    assert_eq!(rt.types().kind(run), &TyKind::Ref(Mutability::Mut, slice));
    assert_eq!(rt.types().kind(slice), &TyKind::Slice(int));
    let pair = rt.declare::<(i64, bool)>();
    let boolean = rt.declare::<bool>();
    assert_eq!(rt.types().kind(pair), &TyKind::Tuple(Box::new([int, boolean]), ()));
    let option = rt.declare::<Option<String>>();
    let string = rt.declare::<String>();
    assert_eq!(rt.types().kind(option), &TyKind::Option(string, ()));
}

#[test]
fn a_uniform_declares_what_it_holds_and_a_stand_in_its_variable() {
    let mut rt = runtime();
    assert_eq!(rt.declare::<U<i64>>(), rt.declare::<i64>());
    let var = rt.declare::<U<Var<2>>>();
    assert_eq!(rt.types().kind(var), &TyKind::Var(2));
}

#[test]
fn a_struct_declares_an_object_in_its_fields_text_order() {
    let mut rt = runtime();
    let point = rt.declare::<Point>();
    let (int, string) = (rt.declare::<i64>(), rt.declare::<String>());
    let names = rt.names();
    let fields = Box::new([(names.intern("a"), int), (names.intern("b"), string)]);
    assert_eq!(rt.types().kind(point), &TyKind::Object { fields, label: () });
}

#[test]
fn an_enum_declares_its_variants_in_their_names_text_order() {
    let mut rt = runtime();
    let shape = rt.declare::<Shape>();
    let (float, string) = (rt.declare::<f64>(), rt.declare::<String>());
    let rect = rt.declare::<(f64, f64)>();
    let labeled = rt.declare_with(|cx| cx.object(&[("name", string), ("width", float)]));
    let names = rt.names();
    let variants = Box::new([
        (names.intern("Empty"), None),
        (names.intern("Labeled"), Some(labeled)),
        (names.intern("Rect"), Some(rect)),
        (names.intern("Square"), Some(float)),
    ]);
    let expected = TyKind::Enum { name: QualifiedRef::root(names.intern("Shape")), variants, label: () };
    assert_eq!(rt.types().kind(shape), &expected);
}

fn marked(rt: &Safe, ty: acvus_extern::Ty) -> Mark {
    match rt.types().kind(ty) {
        TyKind::UserDefined { marked: MarkTerm::Known(mark), .. } => *mark,
        other => panic!("an extern-defined type is declared, not {other:?}"),
    }
}

#[test]
fn a_whole_type_written_natively_is_marked_and_a_uniform_one_is_not() {
    let mut rt = runtime();
    let native = rt.declare::<Stack<String>>();
    assert_eq!(marked(&rt, native), Mark::Marked);
    let uniform = rt.declare::<Stack<U<String>>>();
    assert_eq!(marked(&rt, uniform), Mark::Unmarked);
    let variable = rt.declare::<Stack<U<Var<0>>>>();
    assert_eq!(marked(&rt, variable), Mark::Unmarked);
    let bare = rt.declare::<Counter>();
    assert_eq!(marked(&rt, bare), Mark::Unmarked);
}

#[test]
fn an_extern_defined_type_is_recorded_once_with_its_parameters() {
    let mut rt = runtime();
    rt.declare::<Stack<String>>();
    rt.declare::<Stack<U<String>>>();
    rt.declare::<Cursor<'static, U<Var<0>>>>();
    rt.declare::<Session<EffectAt<3>, IdentityAt<1>>>();
    let recorded: Vec<(String, usize, usize, bool, bool)> = rt
        .user_types()
        .iter()
        .map(|user| {
            let name = rt.names().resolve(user.id.name).to_owned();
            (name, user.type_params.len(), user.lifetimes, user.carries_effect, user.carries_identity)
        })
        .collect();
    let expected = [
        ("Stack".to_owned(), 1, 0, false, false),
        ("Cursor".to_owned(), 1, 1, false, false),
        ("Session".to_owned(), 0, 0, true, true),
    ];
    assert_eq!(recorded, expected);
}

#[test]
fn an_effect_and_an_identity_parameter_are_carried_not_passed_as_arguments() {
    let mut rt = runtime();
    let session = rt.declare::<Session<EffectAt<3>, IdentityAt<1>>>();
    match rt.types().kind(session) {
        TyKind::UserDefined { type_args, effect, identity, .. } => {
            assert!(type_args.is_empty());
            assert_eq!(effect, &Some(EffectTerm::Var(3)));
            assert_eq!(identity, &Some(IdentityTerm::Var(1)));
        }
        other => panic!("an extern-defined type is declared, not {other:?}"),
    }
}
