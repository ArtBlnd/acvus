//! `#[derive(Structural)]`: a struct crosses as an object and an enum as an
//! enum, field by field, each field and variant in its name's text order.
//! Every test ends with no value held.

use acvus_extern::{Crossing, EnumKind, ExternFn, ObjectKind, Registry, Runtime, Structural};
use acvus_extern_test::{Safe, Value, live};
use acvus_mir::decl::{DeclaredFacts, Scheme};
use acvus_mir::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, Poly, TyKind, Types};
use acvus_utils::{Interner, QualifiedRef};

#[derive(Structural, Debug, Clone, PartialEq)]
struct Point {
    b: String,
    a: i64,
}

#[derive(Structural, Debug, Clone, PartialEq)]
struct Pair<T> {
    second: T,
    first: T,
}

#[derive(Structural, Debug, Clone, PartialEq)]
enum Shape {
    Square(f64),
    Empty,
    Rect(f64, f64),
    Labeled { width: f64, name: String },
}

#[derive(Structural, Debug, Clone, PartialEq)]
struct Scene {
    shapes: (Shape, Shape),
    focus: Option<Point>,
    sizes: Result<Pair<i64>, String>,
}

#[derive(Structural, Debug, Clone, PartialEq)]
struct Keyword {
    r#type: bool,
    name: char,
}

fn runtime() -> Safe {
    Safe::new(Interner::new())
}

fn round_trip<T>(rt: &Safe, value: T) -> T
where
    T: Crossing<Safe>,
{
    let given = value.give(rt);
    T::take(rt, given)
}

#[test]
fn a_struct_crosses_as_an_object_in_its_fields_text_order() {
    let rt = runtime();
    let given = Point { b: "x".to_owned(), a: 7 }.give(&rt);
    let [first, second] = <<Safe as Runtime>::Object as ObjectKind<Safe>>::take::<2>(&rt, given);
    assert_eq!(<i64 as Crossing<Safe>>::take(&rt, first), 7);
    assert_eq!(<String as Crossing<Safe>>::take(&rt, second), "x");
    assert_eq!(live(), 0);
}

#[test]
fn a_raw_field_name_is_ordered_by_its_text() {
    let rt = runtime();
    let given = Keyword { r#type: true, name: 'n' }.give(&rt);
    let [first, second] = <<Safe as Runtime>::Object as ObjectKind<Safe>>::take::<2>(&rt, given);
    assert_eq!(<char as Crossing<Safe>>::take(&rt, first), 'n');
    assert!(<bool as Crossing<Safe>>::take(&rt, second));
    assert_eq!(live(), 0);
}

#[test]
fn a_struct_round_trips_generic_or_not() {
    let rt = runtime();
    let point = Point { b: "y".to_owned(), a: -3 };
    assert_eq!(round_trip(&rt, point.clone()), point);
    let pair = Pair { second: 2_i64, first: 1 };
    assert_eq!(round_trip(&rt, pair.clone()), pair);
    assert_eq!(live(), 0);
}

#[test]
fn an_enum_tags_its_variants_in_their_names_text_order() {
    let rt = runtime();
    let tag_of = |shape: Shape| {
        let given = shape.give(&rt);
        let tags: [fn(&Safe, Value) -> usize; 4] = [
            |_, payload| {
                Safe::release(payload);
                0
            },
            |_, payload| {
                Safe::release(payload);
                1
            },
            |_, payload| {
                Safe::release(payload);
                2
            },
            |_, payload| {
                Safe::release(payload);
                3
            },
        ];
        <<Safe as Runtime>::Enum as EnumKind<Safe>>::take(&rt, given, &tags)
    };
    let labeled = Shape::Labeled { width: 1.0, name: "l".to_owned() };
    let tags = [tag_of(Shape::Empty), tag_of(labeled), tag_of(Shape::Rect(1.0, 2.0)), tag_of(Shape::Square(3.0))];
    assert_eq!(tags, [0, 1, 2, 3]);
    assert_eq!(live(), 0);
}

#[test]
fn every_variant_shape_round_trips() {
    let rt = runtime();
    let shapes = [
        Shape::Square(2.5),
        Shape::Empty,
        Shape::Rect(1.0, 4.0),
        Shape::Labeled { width: 3.0, name: "w".to_owned() },
    ];
    for shape in shapes {
        assert_eq!(round_trip(&rt, shape.clone()), shape);
    }
    assert_eq!(live(), 0);
}

#[test]
fn structures_nest_through_tuples_options_and_results() {
    let rt = runtime();
    let scene = Scene {
        shapes: (Shape::Empty, Shape::Rect(2.0, 3.0)),
        focus: Some(Point { b: "f".to_owned(), a: 9 }),
        sizes: Ok(Pair { second: 4, first: 3 }),
    };
    assert_eq!(round_trip(&rt, scene.clone()), scene);
    let bare = Scene { shapes: (Shape::Square(1.0), Shape::Empty), focus: None, sizes: Err("none".to_owned()) };
    assert_eq!(round_trip(&rt, bare.clone()), bare);
    assert_eq!(live(), 0);
}

fn area(shape: Shape) -> f64 {
    match shape {
        Shape::Square(side) => side * side,
        Shape::Empty => 0.0,
        Shape::Rect(width, height) => width * height,
        Shape::Labeled { width, .. } => width * width,
    }
}

/// An entry as the macros will write it, opening its argument by
/// `Crossing` and giving its result by it.
struct Area;

impl<R> ExternFn<R> for Area
where
    R: Runtime,
{
    type Args = [R::Value; 1];

    fn declare(types: &mut Types<Poly>, names: &Interner) -> (QualifiedRef, Scheme) {
        let float = types.intern(TyKind::Float);
        let string = types.intern(TyKind::String);
        let labeled = types.intern(TyKind::Object {
            fields: Box::new([(names.intern("name"), string), (names.intern("width"), float)]),
            label: (),
        });
        let rect = types.intern(TyKind::Tuple(Box::new([float, float]), ()));
        let shape = types.intern(TyKind::Enum {
            name: QualifiedRef::root(names.intern("Shape")),
            variants: Box::new([
                (names.intern("Empty"), None),
                (names.intern("Labeled"), Some(labeled)),
                (names.intern("Rect"), Some(rect)),
                (names.intern("Square"), Some(float)),
            ]),
            label: (),
        });
        let ty = types.intern(TyKind::Fn {
            params: Box::new([shape]),
            ret: float,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let scheme = Scheme {
            ty,
            param_names: Box::new([names.intern("shape")]),
            bounds: Box::new([]),
            impl_of: None,
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
        };
        (QualifiedRef::root(names.intern("area")), scheme)
    }

    fn call(rt: &R, [shape]: [R::Value; 1]) -> R::Value {
        area(Shape::take(rt, shape)).give(rt)
    }
}

#[test]
fn an_extern_fn_takes_an_enum_by_value() {
    let mut rt = runtime();
    rt.register::<Area>();
    let out = rt.call("area", &[Shape::Rect(2.0, 5.0).give(&rt)]);
    assert_eq!(<f64 as Crossing<Safe>>::take(&rt, out), 10.0);
    assert_eq!(live(), 0);
}
