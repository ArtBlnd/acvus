//! The kinds composed, on each runtime: nested values round trip and leave
//! nothing held; generic code crosses through its variables, and one value is
//! read at the generic and a concrete instantiation; a value opens one layer
//! at a time, its parts not yet opened; an exclusive reference changes values
//! in place; a reference owns nothing; an opaque value is read at each
//! instantiation;
//! an option held in its `Value` moves to a box and back as its payload
//! changes; views of a text or a run cross by their kind, alone or nested,
//! and end nothing.

use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::mem::offset_of;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use model::{
    ByMut, ByRef, Calling, CallingMut, CallingOnce, Counted, Cross, Effect0, EffectVar, EnumLayout, EnumTable, Enums, Functions, Generic, ObjectLayout,
    ObjectTable, Objects, Open, Opaques, Programs, Runtime, Scalars, Strings, Structures, TypeVar, Uniform, VariantTable, scope, scope_mut, scope_once,
};

fn round_trip<R, T>(value: T)
where
    R: Counted,
    T: Cross<R> + Clone + PartialEq + Debug,
{
    let before = R::live();
    let given = <T as Cross<R>>::give(value.clone());
    // SAFETY: given just now as `T`, and owned here.
    let taken = unsafe { <T as Cross<R>>::take(given) };
    assert_eq!(taken, value);
    assert_eq!(R::live(), before, "the round trip of {value:?} left a value held");
}

fn nested_values_round_trip<R>()
where
    R: Counted,
{
    round_trip::<R, _>(Some((7i64, String::from("a"))));
    round_trip::<R, _>(None::<(i64, String)>);
    round_trip::<R, _>(Some(Some(3i64)));
    round_trip::<R, _>(Some(None::<i64>));
    round_trip::<R, _>(None::<Option<i64>>);
    round_trip::<R, _>(Some(Some(None::<String>)));
    round_trip::<R, _>(Ok::<Option<String>, (i64, bool)>(Some(String::from("b"))));
    round_trip::<R, _>(Ok::<Option<String>, (i64, bool)>(None));
    round_trip::<R, _>(Err::<Option<String>, (i64, bool)>((1, true)));
    round_trip::<R, _>((Some(1i64), Ok::<String, i64>(String::from("c"))));
    round_trip::<R, _>((None::<i64>, Err::<String, i64>(2)));
    round_trip::<R, _>(Some(Some(Some((String::from("d"), Some(4i64))))));
    round_trip::<R, _>(Some(Some(None::<(String, Option<i64>)>)));
    round_trip::<R, _>(Some(Some(Some((String::from("e"), None::<i64>)))));
}

/// Generic code over a variable, as an extern body is: it gives an
/// `Option<(T, i64)>` through the impl over every variable, beside `i64`'s
/// own.
fn pair_with<R, T>(item: T, count: i64) -> R::Value
where
    R: Runtime,
    T: TypeVar<R>,
{
    <Option<(T, i64)> as Cross<R>>::give(Some((item, count)))
}

/// Generic code takes back what concrete code gave, and the nesting holds.
fn first_of<R, T, U>(value: R::Value) -> Option<T>
where
    R: Runtime,
    T: TypeVar<R>,
    U: TypeVar<R>,
{
    // SAFETY: the caller gives a `Result<Option<(T, U)>, U>` it owns.
    let taken = unsafe { <Result<Option<(T, U)>, U> as Cross<R>>::take(value) };
    taken.ok().flatten().map(|(first, _)| first)
}

fn generic_code_crosses_through_its_variables<R>()
where
    R: Counted,
{
    let before = R::live();
    let item = Uniform::<R, String>::new(String::from("x"));
    // SAFETY: one value at the generic instantiation (variables §3).
    let generic = unsafe { Uniform::<R, Generic>::from_value(item.into_value()) };
    let given = pair_with::<R, _>(generic, 5);
    // SAFETY: given as `Option<(T, i64)>` at the generic instantiation, taken
    // at a concrete one of the same value.
    let taken = unsafe { <Option<(Uniform<R, String>, i64)> as Cross<R>>::take(given) };
    let (item, count) = taken.expect("given as Some");
    assert_eq!(count, 5);
    assert_eq!(item.project(String::clone), "x");
    drop(item);
    assert_eq!(R::live(), before);

    let given = <Result<Option<(String, i64)>, i64> as Cross<R>>::give(Ok(Some((String::from("y"), 6))));
    let first = first_of::<R, Uniform<R, Generic>, Uniform<R, Generic>>(given).expect("given as Ok(Some)");
    // SAFETY: the first part was made as a `String`, read at its concrete
    // instantiation.
    let first = unsafe { Uniform::<R, String>::from_value(first.into_value()) };
    assert_eq!(first.project(String::clone), "y");
    drop(first);
    assert_eq!(R::live(), before);
}

fn a_value_opens_one_layer_at_a_time<R>()
where
    R: Counted,
{
    let before = R::live();
    let held = Uniform::<R, Option<(i64, Option<String>)>>::new(Some((7, Some(String::from("s")))));
    let seen = held.project(|option| {
        let pair = option.as_ref().expect("given as Some");
        pair.project(|(count, text)| {
            let count = count.project(|count| *count);
            let text = text.project(|text| text.as_ref().map(|text| text.project(String::clone)));
            (count, text)
        })
    });
    assert_eq!(seen, (7, Some(String::from("s"))));
    drop(held);

    let held = Uniform::<R, Option<Result<Option<i64>, bool>>>::new(Some(Ok(Some(9))));
    let seen = held.project(|outer| {
        let result = outer.as_ref().expect("given as Some");
        result.project(|result| {
            let Ok(inner) = result else {
                panic!("given as Ok");
            };
            inner.project(|inner| inner.as_ref().map(|number| number.project(|number| *number)))
        })
    });
    assert_eq!(seen, Some(9));
    drop(held);
    assert_eq!(R::live(), before);
}

fn an_exclusive_reference_changes_values_in_place<R>()
where
    R: Counted,
{
    let before = R::live();
    let mut held = Uniform::<R, (i64, Option<String>)>::new((1, None));
    held.project_mut(|(count, text)| {
        count.project_mut(|count| *count = 2);
        text.project_mut(|text| *text = Some(Uniform::new(String::from("u"))));
    });
    let seen = held.project(|(count, text)| {
        (count.project(|count| *count), text.project(|text| text.as_ref().map(|text| text.project(String::clone))))
    });
    assert_eq!(seen, (2, Some(String::from("u"))));
    held.project_mut(|(_, text)| text.project_mut(|text| *text = None));
    // SAFETY: owned here, made as this type.
    let taken = unsafe { <(i64, Option<String>) as Cross<R>>::take(held.into_value()) };
    assert_eq!(taken, (2, None));
    assert_eq!(R::live(), before);
}

fn a_reference_owns_nothing<R>()
where
    R: Counted,
{
    let before = R::live();
    let mut held = Uniform::<R, String>::new(String::from("r"));
    let made = R::live();
    {
        let first = held.as_ref();
        let copy = first;
        let again = first.as_ref();
        assert_eq!(copy.project(String::clone), "r");
        assert_eq!(again.project(String::len), 1);
    }
    {
        let mut exclusive = held.as_mut();
        exclusive.as_mut().project_mut(|text| text.push('!'));
        assert_eq!(exclusive.as_ref().project(String::clone), "r!");
    }
    assert_eq!(R::live(), made, "references made and dropped end no owner");
    drop(held);
    assert_eq!(R::live(), before);
}

/// An extern-defined type, written by hand as `#[opaque(runtime = R,
/// args(T))]` would: put in, taken out, and projected as one value.
struct Stack<T>(Vec<T>);

// SAFETY: the opaque kind takes back what it gave.
unsafe impl<R, T> Cross<R> for Stack<T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    fn give(self) -> R::Value {
        <R::Opaques as Opaques<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Opaques as Opaques<R>>::take(value) }
    }
}

// SAFETY: the opaque kind lends the value itself, by layout.
unsafe impl<'a, R, T> Open<'a, R> for Stack<T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    type Layer = Stack<T>;
    type LayerMut = Stack<T>;

    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Stack<T>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Opaques as Opaques<R>>::project(reference, lent) }
    }

    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Stack<T>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Opaques as Opaques<R>>::project_mut(reference, lent) }
    }
}

/// Generic code over the stack: moves its first item to the end.
fn rotate<R, T>(stack: &mut Uniform<R, Stack<T>>)
where
    R: Runtime,
    T: TypeVar<R>,
{
    stack.project_mut(|stack| {
        let first = stack.0.remove(0);
        stack.0.push(first);
    });
}

fn an_opaque_value_is_read_at_each_instantiation<R>()
where
    R: Counted,
{
    let before = R::live();
    let items = vec![Uniform::<R, String>::new(String::from("a")), Uniform::new(String::from("b"))];
    let stack = Uniform::<R, Stack<Uniform<R, String>>>::new(Stack(items));
    // SAFETY: one value at the generic instantiation (variables §3, §4).
    let mut generic = unsafe { Uniform::<R, Stack<Uniform<R, Generic>>>::from_value(stack.into_value()) };
    rotate::<R, _>(&mut generic);
    // SAFETY: back to its concrete instantiation.
    let stack = unsafe { Uniform::<R, Stack<Uniform<R, String>>>::from_value(generic.into_value()) };
    let texts: Vec<String> = stack.project(|stack| stack.0.iter().map(|item| item.project(String::clone)).collect());
    assert_eq!(texts, ["b", "a"]);
    drop(stack);
    assert_eq!(R::live(), before);
}

fn an_option_changes_how_it_is_held<R>()
where
    R: Counted,
{
    let before = R::live();
    let mut held = Uniform::<R, Option<Option<String>>>::new(Some(None));
    held.project_mut(|outer| {
        let inner = outer.as_mut().expect("given as Some");
        inner.project_mut(|inner| *inner = Some(Uniform::new(String::from("x"))));
    });
    let seen = held.project(|outer| {
        outer.as_ref().and_then(|inner| inner.project(|inner| inner.as_ref().map(|text| text.project(String::clone))))
    });
    assert_eq!(seen, Some(String::from("x")));
    held.project_mut(|outer| outer.as_mut().expect("given as Some").project_mut(|inner| *inner = None));
    // SAFETY: owned here, made as this type.
    let taken = unsafe { <Option<Option<String>> as Cross<R>>::take(held.into_value()) };
    assert_eq!(taken, Some(None));
    assert_eq!(R::live(), before);
}

/// A declared object, written by hand as `#[derive(Structural)]` would: every
/// field a `Uniform` or a variable, its table read off the type by
/// `offset_of!`.
struct Point<R>
where
    R: Runtime,
{
    x: Uniform<R, i64>,
    y: Uniform<R, String>,
}

/// What the derive writes, so that no author writes one: a declared
/// structure's release is its fields' alone, whoever made it.
impl<R> Drop for Point<R>
where
    R: Runtime,
{
    fn drop(&mut self) {}
}

// SAFETY: the table states `Point`'s layout and both fields, each a `Value`.
unsafe impl<R> ObjectLayout for Point<R>
where
    R: Runtime,
{
    const TABLE: &'static ObjectTable = &ObjectTable {
        size: size_of::<Point<R>>(),
        align: align_of::<Point<R>>(),
        fields: &[("x", offset_of!(Point<R>, x)), ("y", offset_of!(Point<R>, y))],
    };
}

// SAFETY: the object kind takes back what it gave.
unsafe impl<R> Cross<R> for Point<R>
where
    R: Runtime,
{
    fn give(self) -> R::Value {
        <R::Objects as Objects<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Objects as Objects<R>>::take(value) }
    }
}

// SAFETY: the object kind lends the object itself, its fields not yet opened.
unsafe impl<'a, R> Open<'a, R> for Point<R>
where
    R: Runtime,
{
    type Layer = Point<R>;
    type LayerMut = Point<R>;

    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Point<R>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Objects as Objects<R>>::project(reference, lent) }
    }

    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Point<R>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Objects as Objects<R>>::project_mut(reference, lent) }
    }
}

/// A declared object over a variable, read at the generic and a concrete
/// instantiation. The runtime is made used by a marker, which is no field.
struct Pair<R, T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    first: T,
    second: T,
    runtime: PhantomData<fn() -> R>,
}

/// As `Point`'s.
impl<R, T> Drop for Pair<R, T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    fn drop(&mut self) {}
}

// SAFETY: as `Point`'s; the marker is zero-sized and holds no `Value`.
unsafe impl<R, T> ObjectLayout for Pair<R, T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    const TABLE: &'static ObjectTable = &ObjectTable {
        size: size_of::<Pair<R, T>>(),
        align: align_of::<Pair<R, T>>(),
        fields: &[("first", offset_of!(Pair<R, T>, first)), ("second", offset_of!(Pair<R, T>, second))],
    };
}

// SAFETY: as `Point`'s.
unsafe impl<R, T> Cross<R> for Pair<R, T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    fn give(self) -> R::Value {
        <R::Objects as Objects<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Objects as Objects<R>>::take(value) }
    }
}

// SAFETY: as `Point`'s.
unsafe impl<'a, R, T> Open<'a, R> for Pair<R, T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    type Layer = Pair<R, T>;
    type LayerMut = Pair<R, T>;

    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Pair<R, T>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Objects as Objects<R>>::project(reference, lent) }
    }

    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Pair<R, T>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Objects as Objects<R>>::project_mut(reference, lent) }
    }
}

/// A declared enum, written by hand as `#[derive(Structural)]` would, its
/// author having written `repr(u32)`: a variant of one unnamed field, one of
/// named fields, and one of none.
#[repr(u32)]
enum Shape<R>
where
    R: Runtime,
{
    Circle(Uniform<R, i64>),
    Rect { w: Uniform<R, i64>, h: Uniform<R, i64> },
    Empty,
}

/// As `Point`'s.
impl<R> Drop for Shape<R>
where
    R: Runtime,
{
    fn drop(&mut self) {}
}

/// Never made: where a `repr(u32)` enum lays out `Circle`'s fields.
#[allow(dead_code)]
#[repr(C)]
struct CircleFields<R>
where
    R: Runtime,
{
    tag: u32,
    radius: Uniform<R, i64>,
}

/// Never made: where a `repr(u32)` enum lays out `Rect`'s fields.
#[allow(dead_code)]
#[repr(C)]
struct RectFields<R>
where
    R: Runtime,
{
    tag: u32,
    w: Uniform<R, i64>,
    h: Uniform<R, i64>,
}

// SAFETY: `Shape` is `repr(u32)`; the tags are its variants' discriminants
// in declaration order, and the fields lie where each variant's `repr(C)`
// struct of the tag and its fields has them.
unsafe impl<R> EnumLayout for Shape<R>
where
    R: Runtime,
{
    const TABLE: &'static EnumTable = &EnumTable {
        size: size_of::<Shape<R>>(),
        align: align_of::<Shape<R>>(),
        variants: &[
            VariantTable { name: "Circle", tag: 0, fields: &[("0", offset_of!(CircleFields<R>, radius))] },
            VariantTable { name: "Rect", tag: 1, fields: &[("w", offset_of!(RectFields<R>, w)), ("h", offset_of!(RectFields<R>, h))] },
            VariantTable { name: "Empty", tag: 2, fields: &[] },
        ],
    };
}

// SAFETY: the enum kind takes back what it gave.
unsafe impl<R> Cross<R> for Shape<R>
where
    R: Runtime,
{
    fn give(self) -> R::Value {
        <R::Enums as Enums<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Enums as Enums<R>>::take(value) }
    }
}

// SAFETY: the enum kind lends the enum itself, its fields not yet opened.
unsafe impl<'a, R> Open<'a, R> for Shape<R>
where
    R: Runtime,
{
    type Layer = Shape<R>;
    type LayerMut = Shape<R>;

    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Shape<R>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Enums as Enums<R>>::project(reference, lent) }
    }

    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Shape<R>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Enums as Enums<R>>::project_mut(reference, lent) }
    }
}

/// The program reads an `i64` field through a reference value the runtime
/// made.
///
/// # Safety
/// `reference` is a live shared or exclusive reference value to an `i64`.
unsafe fn number<R>(reference: R::Value) -> i64
where
    R: Runtime,
{
    // SAFETY: the caller's premise.
    unsafe { <R::Scalars as Scalars<R>>::project::<i64, _>(reference, |number| *number) }
}

fn an_object_the_extern_made_is_read_by_the_runtime<R>()
where
    R: Counted + Structures,
{
    let before = R::live();
    let mut point = Uniform::<R, Point<R>>::new(Point { x: Uniform::new(1), y: Uniform::new(String::from("p")) });
    let shared = point.as_ref().give();
    // SAFETY: a shared reference to a `Point`, read through its table.
    let text = unsafe { <R::Strings as Strings<R>>::project(R::field(shared, "y"), String::clone) };
    assert_eq!(text, "p");
    let exclusive = point.as_mut().give();
    // SAFETY: an exclusive reference to a `Point`, written through its table.
    unsafe { <R::Scalars as Scalars<R>>::project_mut::<i64, _>(R::field(exclusive, "x"), |number| *number = 5) };
    assert_eq!(point.project(|point| point.x.project(|number| *number)), 5);
    drop(point);
    assert_eq!(R::live(), before);
}

fn the_caller_makes_a_declared_object_the_extern_reads<R>()
where
    R: Counted + Structures,
{
    let before = R::live();
    let x = <i64 as Cross<R>>::give(2);
    let y = <String as Cross<R>>::give(String::from("q"));
    // The caller coerces its own object into the declaration's type, making
    // one by the declaration's table (compiler types §1, §3), its fields
    // named in no particular order.
    // SAFETY: both values are live, owned here, and of their fields' types.
    let made = unsafe { R::make_object(<Point<R> as ObjectLayout>::TABLE, &[("y", y), ("x", x)]) };
    // SAFETY: made as a `Point`, owned here.
    let point = unsafe { Uniform::<R, Point<R>>::from_value(made) };
    let seen = point.project(|point| (point.x.project(|number| *number), point.y.project(String::clone)));
    assert_eq!(seen, (2, String::from("q")));
    // SAFETY: owned here, made as a `Point`.
    let taken = unsafe { <Point<R> as Cross<R>>::take(point.into_value()) };
    assert_eq!((taken.x.project(|number| *number), taken.y.project(String::clone)), (2, String::from("q")));
    drop(taken);
    assert_eq!(R::live(), before);
}

/// Generic code over the pair: swaps its fields.
fn swap_pair<R, T>(pair: &mut Uniform<R, Pair<R, T>>)
where
    R: Runtime,
    T: TypeVar<R>,
{
    pair.project_mut(|pair| std::mem::swap(&mut pair.first, &mut pair.second));
}

fn an_object_is_read_at_each_instantiation<R>()
where
    R: Counted + Structures,
{
    let before = R::live();
    let pair = Uniform::<R, Pair<R, Uniform<R, String>>>::new(Pair {
        first: Uniform::new(String::from("a")),
        second: Uniform::new(String::from("b")),
        runtime: PhantomData,
    });
    // SAFETY: one value at the generic instantiation (variables §3, §4).
    let mut generic = unsafe { Uniform::<R, Pair<R, Uniform<R, Generic>>>::from_value(pair.into_value()) };
    swap_pair::<R, _>(&mut generic);
    // SAFETY: the runtime reads the field through the table, whichever
    // instantiation made it.
    let first = unsafe { <R::Strings as Strings<R>>::project(R::field(generic.as_ref().give(), "first"), String::clone) };
    assert_eq!(first, "b");
    // SAFETY: back to its concrete instantiation.
    let pair = unsafe { Uniform::<R, Pair<R, Uniform<R, String>>>::from_value(generic.into_value()) };
    let seen = pair.project(|pair| (pair.first.project(String::clone), pair.second.project(String::clone)));
    assert_eq!(seen, (String::from("b"), String::from("a")));
    drop(pair);
    assert_eq!(R::live(), before);
}

fn an_enum_the_extern_made_is_read_by_the_runtime<R>()
where
    R: Counted + Structures,
{
    let before = R::live();
    let shape = Uniform::<R, Shape<R>>::new(Shape::Rect { w: Uniform::new(2), h: Uniform::new(3) });
    let reference = shape.as_ref().give();
    // SAFETY: a shared reference to a `Shape`, read through its table.
    assert_eq!(unsafe { R::variant(reference) }, "Rect");
    // SAFETY: as above, `h` an `i64`.
    assert_eq!(unsafe { number::<R>(R::field(reference, "h")) }, 3);
    let area = shape.project(|shape| match shape {
        Shape::Rect { w, h } => w.project(|w| *w) * h.project(|h| *h),
        Shape::Circle(_) | Shape::Empty => panic!("made as Rect"),
    });
    assert_eq!(area, 6);
    drop(shape);
    assert_eq!(R::live(), before);
}

fn the_caller_makes_a_declared_enum_the_extern_reads<R>()
where
    R: Counted + Structures,
{
    let before = R::live();
    let radius = <i64 as Cross<R>>::give(4);
    // SAFETY: the value is live, owned here, and of the field's type.
    let made = unsafe { R::make_enum(<Shape<R> as EnumLayout>::TABLE, "Circle", &[("0", radius)]) };
    // SAFETY: made as a `Shape`, owned here.
    let shape = unsafe { Uniform::<R, Shape<R>>::from_value(made) };
    let radius = shape.project(|shape| match shape {
        Shape::Circle(radius) => radius.project(|radius| *radius),
        Shape::Rect { .. } | Shape::Empty => panic!("made as Circle"),
    });
    assert_eq!(radius, 4);
    drop(shape);
    assert_eq!(R::live(), before);
}

fn an_enum_changes_its_variant_in_place<R>()
where
    R: Counted + Structures,
{
    let before = R::live();
    let mut shape = Uniform::<R, Shape<R>>::new(Shape::Circle(Uniform::new(1)));
    shape.project_mut(|shape| *shape = Shape::Rect { w: Uniform::new(5), h: Uniform::new(6) });
    let reference = shape.as_ref().give();
    // SAFETY: a shared reference to a `Shape`, read through its table.
    assert_eq!(unsafe { R::variant(reference) }, "Rect");
    // SAFETY: as above, `w` an `i64`.
    assert_eq!(unsafe { number::<R>(R::field(reference, "w")) }, 5);
    shape.project_mut(|shape| *shape = Shape::Empty);
    // SAFETY: as above.
    assert_eq!(unsafe { R::variant(shape.as_ref().give()) }, "Empty");
    drop(shape);
    assert_eq!(R::live(), before);
}

// Function values. Each program stands for an acvus function the runtime
// holds; each `lend_*` is what an entry writes for a closure parameter of one
// bound, and each body is an extern fn's.

/// `|x| x > 1`, its argument a shared reference to an `i64`.
fn greater_than_one<R>() -> R::Value
where
    R: Programs,
{
    R::function(|arguments| {
        // SAFETY: called with a shared reference to an `i64`.
        let number = unsafe { <R::Scalars as Scalars<R>>::project::<i64, _>(arguments[0], |number| *number) };
        <bool as Cross<R>>::give(number > 1)
    })
}

/// `|x| *x += 1`, its argument an exclusive reference to an `i64`.
fn increment<R>() -> R::Value
where
    R: Programs,
{
    R::function(|arguments| {
        // SAFETY: called with an exclusive reference to an `i64`.
        unsafe { <R::Scalars as Scalars<R>>::project_mut::<i64, _>(arguments[0], |number| *number += 1) };
        <() as Cross<R>>::give(())
    })
}

/// `|text| text.len()`, its argument a `String` it takes over.
fn length<R>() -> R::Value
where
    R: Programs,
{
    R::function(|arguments| {
        // SAFETY: called with a `String` given up to the call.
        let text = unsafe { <String as Cross<R>>::take(arguments[0]) };
        <i64 as Cross<R>>::give(i64::try_from(text.len()).expect("a length fits an i64"))
    })
}

/// `|text| text == "b"`, its argument a shared reference to a `String`.
fn is_b<R>() -> R::Value
where
    R: Programs,
{
    R::function(|arguments| {
        // SAFETY: called with a shared reference to a `String`.
        let found = unsafe { <R::Strings as Strings<R>>::project(arguments[0], |text| text == "b") };
        <bool as Cross<R>>::give(found)
    })
}

/// What an entry writes for `impl Fn(&T) -> bool`: a Rust closure over a
/// shared reference to the function value, its argument given as a `Ref`,
/// its result taken back by its kind.
fn lend_predicate<'f, R, T>(function: &'f R::Value) -> impl Fn(&T) -> bool + 'f
where
    R: Runtime,
    T: TypeVar<R>,
{
    // SAFETY: a shared reference to the function value the call was lent,
    // for the call.
    let function = unsafe { R::value_as_ref(function) };
    move |item: &T| {
        let argument = <model::Ref<'_, R, T> as Cross<R>>::give(item.as_ref());
        // SAFETY: the function takes a shared reference to a `T` and gives a
        // `bool`.
        let answer = unsafe { <R::Functions as Functions<R>>::call_ref(function, [argument]) };
        // SAFETY: as above.
        unsafe { <bool as Cross<R>>::take(answer) }
    }
}

/// What an entry writes for `impl FnMut(&mut T)`.
fn lend_step<'f, R, T>(function: &'f mut R::Value) -> impl FnMut(&mut T) + 'f
where
    R: Runtime,
    T: TypeVar<R>,
{
    // SAFETY: an exclusive reference to the function value the call was
    // lent, for the call.
    let function = unsafe { R::value_as_mut(function) };
    move |item: &mut T| {
        let argument = <model::Mut<'_, R, T> as Cross<R>>::give(item.as_mut());
        // SAFETY: the function takes an exclusive reference to a `T` and
        // gives `()`.
        let answer = unsafe { <R::Functions as Functions<R>>::call_mut(function, [argument]) };
        // SAFETY: as above.
        unsafe { <() as Cross<R>>::take(answer) }
    }
}

/// What an entry writes for `impl FnOnce(String) -> i64`: the closure owns
/// the function value and consumes it.
fn lend_once<R>(function: R::Value) -> impl FnOnce(String) -> i64
where
    R: Runtime,
{
    move |text: String| {
        let argument = <String as Cross<R>>::give(text);
        // SAFETY: the function takes a `String` and gives an `i64`; it is the
        // call's, consumed here.
        let answer = unsafe { <R::Functions as Functions<R>>::call_once(function, [argument]) };
        // SAFETY: as above.
        unsafe { <i64 as Cross<R>>::take(answer) }
    }
}

fn count_matching<T>(items: &[T], predicate: impl Fn(&T) -> bool) -> i64 {
    items.iter().filter(|item| predicate(item)).fold(0, |count, _| count + 1)
}

fn each_mut<T>(items: &mut [T], mut step: impl FnMut(&mut T)) {
    items.iter_mut().for_each(|item| step(item));
}

fn once_with(text: String, finish: impl FnOnce(String) -> i64) -> i64 {
    finish(text)
}

fn closure_parameters_are_called_by_their_bounds<R>()
where
    R: Counted + Programs,
{
    let before = R::live();
    let mut items = [Uniform::<R, i64>::new(1), Uniform::new(2), Uniform::new(3)];

    let predicate = greater_than_one::<R>();
    assert_eq!(count_matching(&items, lend_predicate::<R, Uniform<R, i64>>(&predicate)), 2);

    let mut step = increment::<R>();
    each_mut(&mut items, lend_step::<R, Uniform<R, i64>>(&mut step));
    let numbers = items.each_ref().map(|item| item.project(|number| *number));
    assert_eq!(numbers, [2, 3, 4]);

    assert_eq!(once_with(String::from("abc"), lend_once::<R>(length::<R>())), 3);

    // SAFETY: the caller's function values, released once.
    unsafe {
        R::release(predicate);
        R::release(step);
    }
    drop(items);
    assert_eq!(R::live(), before);
}

/// Generic code calls a closure over its variable.
fn any_match<T>(items: &[T], predicate: impl Fn(&T) -> bool) -> bool {
    items.iter().any(|item| predicate(item))
}

fn generic_code_calls_a_closure_over_its_variable<R>()
where
    R: Counted + Programs,
{
    let before = R::live();
    // SAFETY: each a `String` at the generic instantiation (variables §3).
    let items = [String::from("a"), String::from("b")]
        .map(|text| unsafe { Uniform::<R, Generic>::from_value(Uniform::<R, String>::new(text).into_value()) });
    let predicate = is_b::<R>();
    assert!(any_match(&items, lend_predicate::<R, Uniform<R, Generic>>(&predicate)));
    // SAFETY: the caller's function value, released once.
    unsafe { R::release(predicate) };
    drop(items);
    assert_eq!(R::live(), before);
}

/// `#[fn_type(ref)] type Pred<'f, T, E: EffectVar> = fn(&T) -> bool;`,
/// expanded by hand: a marker carrying the function's effect variable, and a
/// projection in the author's crate that calls it through a shared borrow
/// (functions §2).
struct Pred<'f, T, E>(PhantomData<(fn(T) -> T, fn() -> &'f (), fn(E) -> E)>);

trait PredProjection<R, T>
where
    R: Runtime,
{
    fn call_ref(&self) -> impl Fn(&T) -> bool + '_;
}

impl<'f, R, T, E> PredProjection<R, T> for Uniform<R, Pred<'f, T, E>>
where
    R: Runtime,
    T: TypeVar<R>,
    E: EffectVar,
{
    fn call_ref(&self) -> impl Fn(&T) -> bool + '_ {
        let function = <model::Ref<'_, R, Pred<'f, T, E>> as Cross<R>>::give(Uniform::as_ref(self));
        move |item: &T| {
            let argument = <model::Ref<'_, R, T> as Cross<R>>::give(item.as_ref());
            // SAFETY: a kept `Pred` takes a shared reference to a `T` and
            // gives a `bool`.
            let answer = unsafe { <R::Functions as Functions<R>>::call_ref(function, [argument]) };
            // SAFETY: as above.
            unsafe { <bool as Cross<R>>::take(answer) }
        }
    }
}

/// `#[fn_type(mut)] type Step<'f, T, E: EffectVar> = fn(&mut T);`, expanded:
/// its projection calls it through an exclusive borrow.
struct Step<'f, T, E>(PhantomData<(fn(T) -> T, fn() -> &'f (), fn(E) -> E)>);

trait StepProjection<R, T>
where
    R: Runtime,
{
    fn call_mut(&mut self) -> impl FnMut(&mut T) + '_;
}

impl<'f, R, T, E> StepProjection<R, T> for Uniform<R, Step<'f, T, E>>
where
    R: Runtime,
    T: TypeVar<R>,
    E: EffectVar,
{
    fn call_mut(&mut self) -> impl FnMut(&mut T) + '_ {
        let function = <model::Mut<'_, R, Step<'f, T, E>> as Cross<R>>::give(Uniform::as_mut(self));
        move |item: &mut T| {
            let argument = <model::Mut<'_, R, T> as Cross<R>>::give(item.as_mut());
            // SAFETY: a kept `Step` takes an exclusive reference to a `T`; the
            // exclusive borrow of the owner keeps the function from any other
            // use while the projection lives.
            let answer = unsafe { <R::Functions as Functions<R>>::call_mut(function, [argument]) };
            // SAFETY: as above.
            unsafe { <() as Cross<R>>::take(answer) }
        }
    }
}

/// `#[fn_type(owned)] type Finish<'f, E: EffectVar> = fn(String) -> i64;`,
/// expanded: its projection consumes it.
struct Finish<'f, E>(PhantomData<(fn() -> &'f (), fn(E) -> E)>);

trait FinishProjection<R>
where
    R: Runtime,
{
    fn call_once(self) -> impl FnOnce(String) -> i64;
}

impl<'f, R, E> FinishProjection<R> for Uniform<R, Finish<'f, E>>
where
    R: Runtime,
    E: EffectVar,
{
    fn call_once(self) -> impl FnOnce(String) -> i64 {
        let function = self.into_value();
        move |text: String| {
            let argument = <String as Cross<R>>::give(text);
            // SAFETY: a kept `Finish` takes a `String` and gives an `i64`; it
            // is consumed here.
            let answer = unsafe { <R::Functions as Functions<R>>::call_once(function, [argument]) };
            // SAFETY: as above.
            unsafe { <i64 as Cross<R>>::take(answer) }
        }
    }
}

/// An extern-defined type keeping a function past the call, as an iterator
/// adapter does: its effect is its kept function's (variables §1).
struct Filter<'f, R, T, E>
where
    R: Runtime,
    T: TypeVar<R>,
    E: EffectVar,
{
    predicate: Uniform<R, Pred<'f, T, E>>,
    seen: i64,
}

// SAFETY: the opaque kind takes back what it gave.
unsafe impl<'f, R, T, E> Cross<R> for Filter<'f, R, T, E>
where
    R: Runtime,
    T: TypeVar<R>,
    E: EffectVar,
{
    fn give(self) -> R::Value {
        <R::Opaques as Opaques<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Opaques as Opaques<R>>::take(value) }
    }
}

// SAFETY: the opaque kind lends the value itself, by layout.
unsafe impl<'a, 'f, R, T, E> Open<'a, R> for Filter<'f, R, T, E>
where
    R: Runtime,
    T: TypeVar<R>,
    E: EffectVar,
{
    type Layer = Filter<'f, R, T, E>;
    type LayerMut = Filter<'f, R, T, E>;

    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Self::Layer) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Opaques as Opaques<R>>::project(reference, lent) }
    }

    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Self::LayerMut) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Opaques as Opaques<R>>::project_mut(reference, lent) }
    }
}

/// An extern fn keeping the function it is given.
fn keep<'f, R, T, E>(predicate: Uniform<R, Pred<'f, T, E>>) -> Filter<'f, R, T, E>
where
    R: Runtime,
    T: TypeVar<R>,
    E: EffectVar,
{
    Filter { predicate, seen: 0 }
}

/// A later extern fn calling the kept function through its projection.
fn count_kept<R, T, E>(filter: &mut Filter<'_, R, T, E>, items: &[T]) -> i64
where
    R: Runtime,
    T: TypeVar<R>,
    E: EffectVar,
{
    let predicate = filter.predicate.call_ref();
    let count = items.iter().filter(|item| predicate(item)).fold(0, |count, _| count + 1);
    filter.seen += count;
    count
}

fn kept_functions_are_called_as_their_modes_state<R>()
where
    R: Counted + Programs,
{
    let before = R::live();
    let mut items = [Uniform::<R, i64>::new(1), Uniform::new(2), Uniform::new(3)];

    // acvus gives a function value; the extern keeps it in a `Filter`, which
    // goes back to acvus, and a later call counts through it.
    // SAFETY: the function value acvus gave, owned from here.
    let predicate = unsafe { Uniform::<R, Pred<'static, Uniform<R, i64>, Effect0>>::from_value(greater_than_one::<R>()) };
    let filter = <Filter<'static, R, Uniform<R, i64>, Effect0> as Cross<R>>::give(keep(predicate));
    // SAFETY: given as a `Filter`, owned here.
    let mut filter = unsafe { Uniform::<R, Filter<'static, R, Uniform<R, i64>, Effect0>>::from_value(filter) };
    assert_eq!(filter.project_mut(|filter| count_kept(filter, &items)), 2);
    assert_eq!(filter.project(|filter| filter.seen), 2);

    // SAFETY: as above.
    let mut step = unsafe { Uniform::<R, Step<'static, Uniform<R, i64>, Effect0>>::from_value(increment::<R>()) };
    items.iter_mut().for_each(step.call_mut());
    assert_eq!(items.each_ref().map(|item| item.project(|number| *number)), [2, 3, 4]);

    // SAFETY: as above.
    let finish = unsafe { Uniform::<R, Finish<'static, Effect0>>::from_value(length::<R>()) };
    assert_eq!(finish.call_once()(String::from("abcd")), 4);

    drop((filter, step, items));
    assert_eq!(R::live(), before);
}

// Views of a text or a run. Each body is an extern fn's, and each `*_entry`
// is what an entry writes for it: its arguments taken back, and its result
// given, by their kinds, the call's loans at one lifetime of its own
// (declarations §2).

/// `fn as_str(text: &String) -> &str`, `core::as_str` at `String`, which the
/// caller calls where it coerces a `&String` into a `&str` (compiler:
/// operators §2, §5).
fn as_str(text: &String) -> &str {
    text.as_str()
}

/// The entry of `as_str`: a `&String` is lent by the string kind, and the
/// text the body gives back is a view into the string, which outlives the
/// lending.
///
/// # Safety
/// `argument` is a live shared reference value to a `String`, unchanged while
/// the result lives.
unsafe fn as_str_entry<R>(argument: R::Value) -> R::Value
where
    R: Runtime,
{
    // SAFETY: the caller's premise.
    unsafe { <R::Strings as Strings<R>>::project(argument, |text| <&str as Cross<R>>::give(as_str(text))) }
}

/// `fn first_word(text: &str) -> &str`.
fn first_word(text: &str) -> &str {
    text.split_once(' ').map_or(text, |(first, _)| first)
}

/// # Safety
/// `argument` is a live text view for `'call`.
unsafe fn first_word_entry<'call, R>(argument: R::Value) -> R::Value
where
    R: Runtime,
{
    // SAFETY: the caller's premise.
    let text = unsafe { <&'call str as Cross<R>>::take(argument) };
    <&'call str as Cross<R>>::give(first_word(text))
}

/// `fn split_word(text: &str) -> Option<(&str, &str)>`: views nested in the
/// result, each given by its own kind inside the tuple's and the option's.
fn split_word(text: &str) -> Option<(&str, &str)> {
    text.split_once(' ')
}

/// # Safety
/// As `first_word_entry`.
unsafe fn split_word_entry<'call, R>(argument: R::Value) -> R::Value
where
    R: Runtime,
{
    // SAFETY: the caller's premise.
    let text = unsafe { <&'call str as Cross<R>>::take(argument) };
    <Option<(&'call str, &'call str)> as Cross<R>>::give(split_word(text))
}

fn an_extern_returns_views_of_a_text<R>()
where
    R: Counted,
{
    let before = R::live();
    let held = Uniform::<R, String>::new(String::from("views of text"));
    // The caller coerces its `&String` into the declarations' `&str`.
    // SAFETY: a shared reference to a `String`, held unchanged below.
    let text = unsafe { as_str_entry::<R>(held.as_ref().give()) };
    // SAFETY: a text view of the held string.
    let word = unsafe { first_word_entry::<R>(text) };
    // SAFETY: the result borrows the argument's text, which is held.
    assert_eq!(unsafe { <&str as Cross<R>>::take(word) }, "views");
    // SAFETY: as above.
    let split = unsafe { split_word_entry::<R>(text) };
    // The caller takes the result back as declared; the option and the
    // tuple end with it, and the views they held end nothing.
    // SAFETY: given as this type, owned here, its views into the held text.
    let taken = unsafe { <Option<(&str, &str)> as Cross<R>>::take(split) };
    assert_eq!(taken, Some(("views", "of text")));
    // SAFETY: a view owns nothing, so its release ends nothing.
    unsafe { R::release(text) };
    assert_eq!(held.project(String::clone), "views of text");

    // A literal is a `&str` (compiler: operators §1).
    let literal = <&str as Cross<R>>::give("a literal");
    // SAFETY: a view of a literal, which lives as long as the program.
    assert_eq!(unsafe { <&str as Cross<R>>::take(first_word_entry::<R>(literal)) }, "a");
    // A value holding views releases them with it, ending nothing for them.
    drop(Uniform::<R, Option<(&'static str, i64)>>::new(Some(("held", 1))));
    drop(held);
    assert_eq!(R::live(), before);
}

/// `fn rest<T>(items: &[T]) -> &[T]`, generic.
fn rest<T>(items: &[T]) -> &[T] {
    items.split_first().map_or(items, |(_, rest)| rest)
}

/// `fn tail_mut<T>(items: &mut [T]) -> &mut [T]`, generic.
fn tail_mut<T>(items: &mut [T]) -> &mut [T] {
    let from = items.len().min(1);
    &mut items[from..]
}

/// The entry of `rest`, at the generic instantiation (variables §3).
///
/// # Safety
/// `argument` is a live run view for `'call`.
unsafe fn rest_entry<'call, R>(argument: R::Value) -> R::Value
where
    R: Runtime,
{
    // SAFETY: the caller's premise.
    let items = unsafe { <&'call [Uniform<R, Generic>] as Cross<R>>::take(argument) };
    <&'call [Uniform<R, Generic>] as Cross<R>>::give(rest(items))
}

/// The entry of `tail_mut`, at the generic instantiation.
///
/// # Safety
/// `argument` is a live exclusive run view for `'call`.
unsafe fn tail_mut_entry<'call, R>(argument: R::Value) -> R::Value
where
    R: Runtime,
{
    // SAFETY: the caller's premise.
    let items = unsafe { <&'call mut [Uniform<R, Generic>] as Cross<R>>::take(argument) };
    <&'call mut [Uniform<R, Generic>] as Cross<R>>::give(tail_mut(items))
}

/// The entry of `fn reverse<T>(items: &mut [T])`, at the generic instantiation.
///
/// # Safety
/// As `tail_mut_entry`.
unsafe fn reverse_entry<'call, R>(argument: R::Value) -> R::Value
where
    R: Runtime,
{
    // SAFETY: the caller's premise.
    let items = unsafe { <&'call mut [Uniform<R, Generic>] as Cross<R>>::take(argument) };
    items.reverse();
    <() as Cross<R>>::give(())
}

fn numbers<R>(run: &[Uniform<R, i64>]) -> Vec<i64>
where
    R: Runtime,
{
    run.iter().map(|item| item.project(|number| *number)).collect()
}

fn an_extern_returns_views_of_a_run<R>()
where
    R: Counted,
{
    let before = R::live();
    // A run of values, laid out as a container the runtime holds lays out
    // its elements.
    let mut items: Vec<Uniform<R, i64>> = (1..=4).map(Uniform::new).collect();
    // SAFETY: a view of the run, unchanged while the result is read.
    let rest = unsafe { rest_entry::<R>(<&[Uniform<R, i64>] as Cross<R>>::give(items.as_slice())) };
    // SAFETY: the result borrows the run, read at its concrete instantiation.
    assert_eq!(numbers(unsafe { <&[Uniform<R, i64>] as Cross<R>>::take(rest) }), [2, 3, 4]);
    // SAFETY: an exclusive view of the run, reached through nothing else
    // while the result is used.
    let tail = unsafe { tail_mut_entry::<R>(<&mut [Uniform<R, i64>] as Cross<R>>::give(items.as_mut_slice())) };
    // SAFETY: as above.
    let tail = unsafe { <&mut [Uniform<R, i64>] as Cross<R>>::take(tail) };
    tail.iter_mut().for_each(|item| item.project_mut(|number| *number *= 10));
    // SAFETY: as above.
    let done = unsafe { reverse_entry::<R>(<&mut [Uniform<R, i64>] as Cross<R>>::give(items.as_mut_slice())) };
    // SAFETY: the entry gave `()`.
    unsafe { <() as Cross<R>>::take(done) };
    assert_eq!(numbers(&items), [40, 30, 20, 1]);
    drop(items);
    assert_eq!(R::live(), before);
}

// Asynchronous calls (functions §3). The call runs in place, as the future
// that awaits it is polled.

/// Polls `future` to its end.
fn block_on<O>(future: impl Future<Output = O>) -> O {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(out) = future.as_mut().poll(&mut context) {
            return out;
        }
    }
}

/// Polls `future` once.
fn poll_once<F>(future: Pin<&mut F>) -> Poll<F::Output>
where
    F: Future,
{
    future.poll(&mut Context::from_waker(Waker::noop()))
}

/// Suspends once.
struct YieldNow(bool);

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        match self.0 {
            true => Poll::Ready(()),
            false => {
                self.0 = true;
                Poll::Pending
            }
        }
    }
}

/// `async |counter, text| { wait `yields` times; *counter += 1; text.len() }`,
/// its arguments an exclusive reference to an `i64` and a shared one to a
/// `String`.
fn counting<R>(yields: usize) -> R::Value
where
    R: Programs,
{
    R::async_function(move |arguments| {
        let (counter, text) = (arguments[0], arguments[1]);
        Box::pin(async move {
            for _ in 0..yields {
                YieldNow(false).await;
            }
            // SAFETY: called with an exclusive reference to an `i64`, which
            // the scope keeps for the call.
            unsafe { <R::Scalars as Scalars<R>>::project_mut::<i64, _>(counter, |count| *count += 1) };
            // SAFETY: called with a shared reference to a `String`, as above.
            let length = unsafe { <R::Strings as Strings<R>>::project(text, String::len) };
            <i64 as Cross<R>>::give(i64::try_from(length).expect("a length fits an i64"))
        })
    })
}

/// `#[fn_type(async)] type Count<'f, C, S, E: EffectVar> = fn(&mut C, &S) -> i64;`,
/// expanded by hand: a marker carrying the function's effect variable, and
/// the call a scope lends it to, in the author's crate.
struct Count<'f, C, S, E>(PhantomData<(fn(C) -> C, fn(S) -> S, fn() -> &'f (), fn(E) -> E)>);

trait CountCall<'s, R, C, S>
where
    R: Runtime,
{
    fn call(self, counter: model::Mut<'s, R, C>, text: model::Ref<'s, R, S>) -> impl Future<Output = i64> + 's;
}

impl<'s, 'f, R, C, S, E> CountCall<'s, R, C, S> for Calling<'s, R, Count<'f, C, S, E>>
where
    R: Runtime,
    C: TypeVar<R>,
    S: TypeVar<R>,
    E: EffectVar,
{
    fn call(self, counter: model::Mut<'s, R, C>, text: model::Ref<'s, R, S>) -> impl Future<Output = i64> + 's {
        let arguments = [<model::Mut<'s, R, C> as Cross<R>>::give(counter), <model::Ref<'s, R, S> as Cross<R>>::give(text)];
        // SAFETY: a kept `Count` takes an exclusive reference to a `C` and a
        // shared one to an `S`; the scope keeps the function and both for
        // `'s`, which the call does not outlive.
        let call = unsafe { <R::Functions as Functions<R>>::call_async_ref(self.into_value(), arguments) };
        async move {
            let answer = call.await;
            // SAFETY: a kept `Count` gives an `i64`.
            unsafe { <i64 as Cross<R>>::take(answer) }
        }
    }
}

type Counter<R> = Count<'static, Uniform<R, i64>, Uniform<R, String>, Effect0>;

/// The function value acvus gave, and the arguments the extern lends it.
fn counter_and_arguments<R>(yields: usize) -> (Uniform<R, Counter<R>>, (ByMut<Uniform<R, i64>>, ByRef<Uniform<R, String>>))
where
    R: Programs,
{
    // SAFETY: the function value acvus gave, owned from here.
    let function = unsafe { Uniform::<R, Counter<R>>::from_value(counting::<R>(yields)) };
    (function, (ByMut(Uniform::new(1)), ByRef(Uniform::new(String::from("abc")))))
}

fn a_scope_gives_its_arguments_back_with_the_result<R>()
where
    R: Counted + Programs,
{
    let before = R::live();
    let (function, arguments) = counter_and_arguments::<R>(2);
    let (function, (ByMut(counter), ByRef(text)), out) =
        block_on(scope(function, arguments, async |calling, (counter, text)| calling.call(counter, text).await));
    assert_eq!(out, 3);
    assert_eq!(counter.project(|count| *count), 2);
    assert_eq!(text.project(String::clone), "abc");
    drop((function, counter, text));
    assert_eq!(R::live(), before);
}

/// Dropping the scope while its call is suspended abandons the call there:
/// nothing runs it apart, and what the scope holds is released with it.
fn a_cancelled_scope_abandons_its_call<R>()
where
    R: Counted + Programs,
{
    let before = R::live();
    let (function, arguments) = counter_and_arguments::<R>(3);
    {
        let mut pending = std::pin::pin!(scope(function, arguments, async |calling, (counter, text)| calling.call(counter, text).await));
        assert!(poll_once(pending.as_mut()).is_pending());
    }
    assert_eq!(R::live(), before);
}

/// `async |counter| { wait once; calls += 1; *counter += calls; calls }`: a
/// function value whose own state, `calls`, each call changes, so it is
/// called through an exclusive reference; its argument an exclusive
/// reference to an `i64`.
fn tallying<R>() -> R::Value
where
    R: Programs,
{
    let mut calls = 0i64;
    R::async_function(move |arguments| {
        calls += 1;
        let (counter, calls) = (arguments[0], calls);
        Box::pin(async move {
            YieldNow(false).await;
            // SAFETY: called with an exclusive reference to an `i64`, which
            // the scope keeps for the call.
            unsafe { <R::Scalars as Scalars<R>>::project_mut::<i64, _>(counter, |count| *count += calls) };
            <i64 as Cross<R>>::give(calls)
        })
    })
}

/// `#[fn_type(mut, async)] type Tally<'f, C, E: EffectVar> = fn(&mut C) -> i64;`,
/// expanded by hand: called asynchronously through an exclusive reference.
struct Tally<'f, C, E>(PhantomData<(fn(C) -> C, fn() -> &'f (), fn(E) -> E)>);

trait TallyCall<'s, R, C>
where
    R: Runtime,
{
    fn call(self, counter: model::Mut<'s, R, C>) -> impl Future<Output = i64> + 's;
}

impl<'s, 'f, R, C, E> TallyCall<'s, R, C> for CallingMut<'s, R, Tally<'f, C, E>>
where
    R: Runtime,
    C: TypeVar<R>,
    E: EffectVar,
{
    fn call(self, counter: model::Mut<'s, R, C>) -> impl Future<Output = i64> + 's {
        let arguments = [<model::Mut<'s, R, C> as Cross<R>>::give(counter)];
        // SAFETY: a kept `Tally` takes an exclusive reference to a `C`; the
        // scope keeps the function, reached through this call alone, and the
        // argument for `'s`, which the call does not outlive.
        let call = unsafe { <R::Functions as Functions<R>>::call_async_mut(self.into_value(), arguments) };
        async move {
            let answer = call.await;
            // SAFETY: a kept `Tally` gives an `i64`.
            unsafe { <i64 as Cross<R>>::take(answer) }
        }
    }
}

/// `async |text| { wait once; text.len() }`, its argument a shared reference
/// to a `String`; called by value, once.
fn measuring<R>() -> R::Value
where
    R: Programs,
{
    R::async_function(|arguments| {
        let text = arguments[0];
        Box::pin(async move {
            YieldNow(false).await;
            // SAFETY: called with a shared reference to a `String`, which the
            // scope keeps for the call.
            let length = unsafe { <R::Strings as Strings<R>>::project(text, String::len) };
            <i64 as Cross<R>>::give(i64::try_from(length).expect("a length fits an i64"))
        })
    })
}

/// `#[fn_type(owned, async)] type Measure<'f, S, E: EffectVar> = fn(&S) -> i64;`,
/// expanded by hand: called asynchronously by value, which consumes it.
struct Measure<'f, S, E>(PhantomData<(fn(S) -> S, fn() -> &'f (), fn(E) -> E)>);

trait MeasureCall<'s, R, S>
where
    R: Runtime,
{
    fn call(self, text: model::Ref<'s, R, S>) -> impl Future<Output = i64> + 's;
}

impl<'s, 'f, R, S, E> MeasureCall<'s, R, S> for CallingOnce<'s, R, Measure<'f, S, E>>
where
    R: Runtime,
    S: TypeVar<R>,
    E: EffectVar,
{
    fn call(self, text: model::Ref<'s, R, S>) -> impl Future<Output = i64> + 's {
        let arguments = [<model::Ref<'s, R, S> as Cross<R>>::give(text)];
        // SAFETY: a kept `Measure` takes a shared reference to an `S`; the
        // function is given up to the call, and the scope keeps the argument
        // for `'s`, which the call does not outlive.
        let call = unsafe { <R::Functions as Functions<R>>::call_async_once(self.into_value(), arguments) };
        async move {
            let answer = call.await;
            // SAFETY: a kept `Measure` gives an `i64`.
            unsafe { <i64 as Cross<R>>::take(answer) }
        }
    }
}

type Tallier<R> = Tally<'static, Uniform<R, i64>, Effect0>;
type Measurer<R> = Measure<'static, Uniform<R, String>, Effect0>;

/// An exclusive scope lends its call the function to change, and gives it
/// back changed: a second scope's call sees what the first did.
fn an_exclusive_scope_calls_its_function_in_place<R>()
where
    R: Counted + Programs,
{
    let before = R::live();
    // SAFETY: the function value acvus gave, owned from here.
    let function = unsafe { Uniform::<R, Tallier<R>>::from_value(tallying::<R>()) };
    let (function, ByMut(counter), first) =
        block_on(scope_mut(function, ByMut(Uniform::<R, i64>::new(0)), async |calling, counter| calling.call(counter).await));
    assert_eq!((first, counter.project(|count| *count)), (1, 1));
    let (function, ByMut(counter), second) = block_on(scope_mut(function, ByMut(counter), async |calling, counter| calling.call(counter).await));
    assert_eq!((second, counter.project(|count| *count)), (2, 3));
    drop((function, counter));
    assert_eq!(R::live(), before);
}

/// An owning scope gives its call the function, which the call consumes:
/// only the arguments and the result come back, and the function is released
/// when the call ends.
fn an_owning_scope_gives_its_function_to_the_call<R>()
where
    R: Counted + Programs,
{
    let before = R::live();
    // SAFETY: as above.
    let function = unsafe { Uniform::<R, Measurer<R>>::from_value(measuring::<R>()) };
    let text = Uniform::<R, String>::new(String::from("abc"));
    let with_text = R::live();
    let (ByRef(text), length) = block_on(scope_once(function, ByRef(text), async |calling, text| calling.call(text).await));
    assert_eq!(length, 3);
    assert_eq!(R::live(), with_text - 1, "the call released the function it consumed");
    drop(text);
    assert_eq!(R::live(), before);
}

/// Dropping an exclusive or an owning scope while its call is suspended
/// abandons the call there, and releases what the scope holds: the function
/// lent exclusively, or the function the call owns.
fn a_cancelled_exclusive_or_owning_scope_releases_what_it_holds<R>()
where
    R: Counted + Programs,
{
    let before = R::live();
    // SAFETY: as above.
    let function = unsafe { Uniform::<R, Tallier<R>>::from_value(tallying::<R>()) };
    {
        let mut pending =
            std::pin::pin!(scope_mut(function, ByMut(Uniform::<R, i64>::new(0)), async |calling, counter| calling.call(counter).await));
        assert!(poll_once(pending.as_mut()).is_pending());
    }
    assert_eq!(R::live(), before);

    // SAFETY: as above.
    let function = unsafe { Uniform::<R, Measurer<R>>::from_value(measuring::<R>()) };
    {
        let text = Uniform::<R, String>::new(String::from("abc"));
        let mut pending = std::pin::pin!(scope_once(function, ByRef(text), async |calling, text| calling.call(text).await));
        assert!(poll_once(pending.as_mut()).is_pending());
    }
    assert_eq!(R::live(), before);

    // A call never made: the callable the owning scope gave is released with
    // the work that held it.
    // SAFETY: as above.
    let function = unsafe { Uniform::<R, Measurer<R>>::from_value(measuring::<R>()) };
    {
        let text = Uniform::<R, String>::new(String::from("abc"));
        let mut pending = std::pin::pin!(scope_once(function, ByRef(text), async |calling, text| {
            YieldNow(false).await;
            calling.call(text).await
        }));
        assert!(poll_once(pending.as_mut()).is_pending());
    }
    assert_eq!(R::live(), before);
}

/// Generic code reborrowing its variable's value exclusively, and giving
/// out the reference value it gets.
fn exclusive_reborrow<R, T>(item: &mut T) -> R::Value
where
    R: Runtime,
    T: TypeVar<R>,
{
    <model::Mut<'_, R, T> as Cross<R>>::give(item.as_mut())
}

/// A reference to a reference is that reference (references §1): an
/// exclusive reborrow of a variable whose value is a shared reference gives
/// that shared reference.
fn an_exclusive_reborrow_of_a_shared_reference_is_that_reference<R>()
where
    R: Counted,
{
    let before = R::live();
    let owner = Uniform::<R, String>::new(String::from("owner"));
    let shared = <model::Ref<'_, R, String> as Cross<R>>::give(owner.as_ref());
    // SAFETY: a shared reference to `owner` as a variable's value, at the
    // generic instantiation (variables §3).
    let mut generic = unsafe { Uniform::<R, Generic>::from_value(shared) };
    let reference = exclusive_reborrow::<R, _>(&mut generic);
    // SAFETY: the reference value is the shared reference `generic` holds, to
    // `owner`, which is not changed while it is read.
    assert_eq!(unsafe { model::Ref::<R, String>::from_value(reference) }.project(String::clone), "owner");
    drop(generic);
    assert_eq!(owner.project(String::clone), "owner");
    drop(owner);
    assert_eq!(R::live(), before);
}

macro_rules! on_every_runtime {
    ($($test:ident),* $(,)?) => {
        mod slots {
            $(
                #[test]
                fn $test() {
                    super::$test::<model::slots::Slots>();
                }
            )*
        }

        mod boxes_on_the_heap {
            $(
                #[test]
                fn $test() {
                    super::$test::<model::boxes::Boxes<model::boxes::Heap, model::boxes::Unchained>>();
                }
            )*
        }

        mod chained_boxes_on_the_heap {
            $(
                #[test]
                fn $test() {
                    super::$test::<model::boxes::Boxes<model::boxes::Heap, model::boxes::Chained>>();
                }
            )*
        }

        mod chained_boxes_in_a_frame {
            $(
                #[test]
                fn $test() {
                    super::$test::<model::boxes::Boxes<model::boxes::Frame, model::boxes::Chained>>();
                }
            )*
        }
    };
}

on_every_runtime! {
    nested_values_round_trip,
    generic_code_crosses_through_its_variables,
    a_value_opens_one_layer_at_a_time,
    an_exclusive_reference_changes_values_in_place,
    a_reference_owns_nothing,
    an_opaque_value_is_read_at_each_instantiation,
    an_option_changes_how_it_is_held,
    an_object_the_extern_made_is_read_by_the_runtime,
    the_caller_makes_a_declared_object_the_extern_reads,
    an_object_is_read_at_each_instantiation,
    an_enum_the_extern_made_is_read_by_the_runtime,
    the_caller_makes_a_declared_enum_the_extern_reads,
    an_enum_changes_its_variant_in_place,
    closure_parameters_are_called_by_their_bounds,
    generic_code_calls_a_closure_over_its_variable,
    kept_functions_are_called_as_their_modes_state,
    an_extern_returns_views_of_a_text,
    an_extern_returns_views_of_a_run,
    a_scope_gives_its_arguments_back_with_the_result,
    a_cancelled_scope_abandons_its_call,
    an_exclusive_scope_calls_its_function_in_place,
    an_owning_scope_gives_its_function_to_the_call,
    a_cancelled_exclusive_or_owning_scope_releases_what_it_holds,
    an_exclusive_reborrow_of_a_shared_reference_is_that_reference,
}
