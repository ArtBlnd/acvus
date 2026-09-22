//! The boundary's one crossing (RFC-0039): every type an ExternFn takes or
//! returns implements `Cross`, and the glue calls it and nothing else. A
//! scalar and an extension type are stored as themselves; a derived struct
//! or enum is rebuilt field by field (RFC-0032, RFC-0036); a container
//! crosses each element; a carrier (`Ref`, `Closure`) is the runtime value it
//! holds. `Obj<V>` and `Variant<V>` are the runtime's own object and
//! variant shapes.

use std::any::{Any, TypeId};
use std::mem::ManuallyDrop;
use std::sync::Arc;

use acvus_utils::{Astr, Interner};

use crate::handler::Uniform;
use crate::len::Arr;
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::ty_arg::{Never, Var, kind};

/// A position in an object's flat layout: which of its type's fields, in the
/// order rule 8 fixes.
///
/// An operation that writes a field holds this beside the frame register it
/// reads the value from, and the two are different spaces of small unsigned
/// numbers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct FieldAt(u16);

/// `FieldAt` names a position in a `u16`, so the checker's bound on an object
/// type's width is that `u16`'s range. The two are one number, and this is what
/// fails if either moves.
const _: () = assert!(
    acvus_mir::ty::ObjectTy::<acvus_mir::ty::Concrete>::MAX_FIELDS == u16::MAX as usize,
    "the checker's object width bound is the range of the u16 FieldAt names a position in"
);

impl FieldAt {
    /// Every position a shape has fits, so this converts without asking: an
    /// object type wider than `ObjectTy::MAX_FIELDS` is refused where its
    /// width is made — an object literal, and the union two field sets join to.
    pub fn of(at: usize) -> FieldAt {
        debug_assert!(
            at <= acvus_mir::ty::ObjectTy::<acvus_mir::ty::Concrete>::MAX_FIELDS,
            "an object is no wider than the checker admits"
        );
        FieldAt(at as u16)
    }

    pub fn index(self) -> usize {
        usize::from(self.0)
    }
}

/// The field names of an object type, in the one order RFC-0050 rule 8 fixes:
/// ascending by the resolved name. One `ObjectShape` is shared by every object of
/// the type, so an object's own allocation is its field values alone and a
/// field is a position in this list.
///
/// Rule 8 also gives a `Declared` struct its declaration's order, and that
/// order does not exist to be read: `ObjectTy` carries its fields as an
/// `FxHashMap` and `FieldSet::Declared` carries the struct's name, not its
/// field order. So string order is the order of every object type, declared
/// or not, which is already the order a committed object's canonical bytes
/// take (`interpreter::layout::sorted_fields`).
pub struct ObjectShape {
    names: Box<[Astr]>,
}

impl ObjectShape {
    /// `names` in rule 8's order.
    pub fn of<I>(interner: &Interner, names: I) -> Arc<ObjectShape>
    where
        I: IntoIterator<Item = Astr>,
    {
        let mut names: Vec<Astr> = names.into_iter().collect();
        names.sort_by(|a, b| interner.resolve(*a).cmp(interner.resolve(*b)));
        Arc::new(ObjectShape {
            names: names.into_boxed_slice(),
        })
    }

    /// `names` already in rule 8's order, which the caller sorted by the same
    /// comparison `of` makes.
    ///
    /// The one caller outside a test is `object_in_order`, whose names come
    /// from `acvus-extern-macro`'s derived field table, sorted over the field
    /// names as string literals at expansion.
    /// `acvus-extern/tests/owned_holders.rs::
    /// a_derived_structs_field_table_is_the_shape_order` pins the two orders
    /// against each other.
    pub fn in_order(names: Box<[Astr]>) -> Arc<ObjectShape> {
        Arc::new(ObjectShape { names })
    }

    pub fn names(&self) -> &[Astr] {
        &self.names
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The position `name`'s field holds. The machine never asks — `prepare`
    /// reads the position off the settled type — and this is the untyped path
    /// rule 6 leaves a handler that needs a field by name at run time.
    pub fn at(&self, name: Astr) -> Option<FieldAt> {
        self.names
            .iter()
            .position(|held| *held == name)
            .map(FieldAt::of)
    }
}

impl PartialEq for ObjectShape {
    fn eq(&self, other: &Self) -> bool {
        self.names == other.names
    }
}

impl Eq for ObjectShape {}

impl std::fmt::Debug for ObjectShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.names.iter()).finish()
    }
}

/// An object as the runtime holds it: its type's field names, shared, and the
/// flat run of its fields' values behind one header (RFC-0050 rules 4 and 8).
/// `values[i]` is the field `shape.names()[i]` names.
pub struct Obj<V> {
    pub shape: Arc<ObjectShape>,
    pub values: Box<[V]>,
}

impl<V> Obj<V> {
    /// One value per field of `shape`, taken at each position in turn, so an
    /// object's width is its shape's without a length to compare.
    pub fn filled<F>(shape: Arc<ObjectShape>, at: F) -> Obj<V>
    where
        F: FnMut(FieldAt) -> V,
    {
        let values = (0..shape.len()).map(FieldAt::of).map(at).collect();
        Obj { shape, values }
    }

    /// `filled` is the constructor with no width to get wrong.
    pub fn new(shape: Arc<ObjectShape>, values: Box<[V]>) -> Obj<V> {
        debug_assert_eq!(
            shape.len(),
            values.len(),
            "an object holds one value per field of its shape"
        );
        Obj { shape, values }
    }

    /// The fields by name, in the layout's order.
    pub fn fields(&self) -> impl Iterator<Item = (Astr, &V)> + '_ {
        self.shape.names().iter().copied().zip(self.values.iter())
    }
}

/// A variant as the runtime holds it (RFC-0050 rules 4 and 8).
///
/// These two registers are the same two that `interpreter::prepare::runs::
/// Layout` gives a run of the same enum, in the same order and with the same
/// word spelling the tag. The two crates lay one enum, and
/// `interpreter::prepare::runs`'s
/// `a_heap_variant_and_a_run_of_one_enum_are_the_same_words` is where the two
/// are compared register for register.
///
/// Unlike its sibling `Obj`, a variant carries no shared shape. A field's
/// position means nothing without the field names, so an object has to hold an
/// `ObjectShape`; a tag word carries the interned name it stands for, so a
/// reader — a printer with no `Ty`, a crossing, a `Switch` — resolves it alone.
pub struct Variant<V> {
    pub values: [V; 2],
}

impl<V> Variant<V> {
    /// The registers rule 8 gives every enum, read off the layout above.
    pub const WIDTH: usize = Variant::of((), ()).values.len();

    pub const fn of(tag: V, payload: V) -> Variant<V> {
        Variant {
            values: [tag, payload],
        }
    }

    pub fn tag(&self) -> &V {
        &self.values[0]
    }

    pub fn payload(&self) -> &V {
        &self.values[1]
    }

    pub fn payload_mut(&mut self) -> &mut V {
        &mut self.values[1]
    }

    pub fn into_payload(self) -> V {
        let [_, payload] = self.values;
        payload
    }
}

/// The run of the runtime's values a crossing occupies, as a type, so that a
/// bound can name it and the width is read off the type rather than repeated
/// as a number.
pub trait Form {
    const WIDTH: usize;

    /// Two forms share a width: a struct of two fields and a view are both
    /// two of the runtime's values, and they land by different rules. This
    /// is what a caller matches on.
    const KIND: FormKind;

    /// `Run` with one more parameter of this form on it, which is how a
    /// declaration's call form is folded out of its parameter list
    /// (`handler::ArgRun`).
    type Onto<Run>: crate::handler::ArgRun
    where
        Run: crate::handler::ArgRun;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormKind {
    Value,
    View,
    /// `Form::WIDTH` of them, one per register of the aggregate's layout.
    Components,
}

/// Obligation across artifacts: a parameter of this form takes its datum
/// from `Sited::Site` instead, so `Parameters::sites` must give it one.
pub struct Nothing;

/// One of the runtime's values.
pub struct One;

/// The two registers the machine keeps a slice in (RFC-0047 amended).
pub struct Pair;

/// An aggregate's `W` components, written where the caller placed its
/// destination run (RFC-0050 rules 5, 6 and 8). `W` is the flat width of the
/// aggregate — one register per field for every struct that reaches a run,
/// since `prepare/runs.rs::Layout::lowerable` refuses any object with a
/// nested aggregate field.
pub struct Run<const W: usize>;

/// A form whose parameter still names what it named after the caller
/// suspends: one of the runtime's values, or nothing of the run at all. A
/// `Pair` borrows the frame the call laid its arguments on, and that frame
/// is gone by the time an awaited call resumes (RFC-0047 §3).
#[diagnostic::on_unimplemented(
    message = "a parameter of form `{Self}` does not survive the caller suspending, so a declaration above `Task::Sync` cannot take one: it borrows the frame the call laid its arguments on, and that frame is gone when the call resumes (RFC-0047 §3)"
)]
pub trait SurvivesSuspension: Form {}

impl SurvivesSuspension for Nothing {}
impl SurvivesSuspension for One {}

impl Form for Nothing {
    const WIDTH: usize = 0;
    const KIND: FormKind = FormKind::Value;

    type Onto<Run>
        = Run
    where
        Run: crate::handler::ArgRun;
}

impl Form for One {
    const WIDTH: usize = 1;
    const KIND: FormKind = FormKind::Value;

    type Onto<Run>
        = <Run as crate::handler::ArgRun>::WithOne
    where
        Run: crate::handler::ArgRun;
}

impl Form for Pair {
    const WIDTH: usize = 2;
    const KIND: FormKind = FormKind::View;

    type Onto<Run>
        = <Run as crate::handler::ArgRun>::WithPair
    where
        Run: crate::handler::ArgRun;
}

impl<const W: usize> Form for Run<W> {
    const WIDTH: usize = W;
    const KIND: FormKind = FormKind::Components;

    /// The fold over `W` register steps is not written, and that is a
    /// decision. No `Arg` impl names `Run<W>`: a by-value aggregate
    /// parameter crosses as the one value rule 4 realizes it into, so
    /// `Cross::Form` is `One` for every aggregate and this association is
    /// never projected. A parameter that did reach it is lent its window.
    type Onto<Run>
        = crate::handler::InWindow
    where
        Run: crate::handler::ArgRun;
}

/// How a type crosses the boundary (RFC-0039): as the run of the runtime's
/// values it occupies. Every type an ExternFn takes or returns implements
/// this. The crossing that is one value is `OneValue`, which every one of
/// them but a slice also implements.
pub trait Cross<Rt>: Sized + Send + Sync + 'static
where
    Rt: Runtime,
{
    type Form: Form;

    /// The run the same type occupies at the return position, which is not
    /// always `Form`: a `#[derive(TyArg)]` struct is one heap object as a
    /// field, a container's element and a by-value parameter, and its own
    /// components as a result (RFC-0050 rules 5 and 6).
    type ReturnForm: Form;

    /// How many of the runtime's values one `Self` occupies. A declaration's
    /// slot count is the sum of its parameters' widths, and the library adds
    /// them from these constants (RFC-0050 rule 6).
    const WIDTH: usize = <Self::Form as Form>::WIDTH;

    /// `Self` read out of the run of `WIDTH` values it was written into.
    ///
    /// # Safety
    /// `run` is `WIDTH` long and holds what `into_run` wrote.
    unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self;

    /// `Self` written into a run of `WIDTH` values.
    fn into_run(self, rt: &Rt, out: &mut [Rt::Value]);

    fn into_return_run(self, rt: &Rt, out: &mut [Rt::Value]) {
        self.into_run(rt, out)
    }
}

/// The crossing of a type that is one of the runtime's values, at one of the
/// two representations a slot can take (`Uniform`, `Specialized` — RFC-0040):
/// `erase` hands the runtime that value, `materialize` takes it back, and
/// `deref` reads a `Self` through a reference the runtime holds, which only a
/// type stored as itself can do. Every bound that needs a value — a
/// parameter, an object's field, a container's element — says this and not
/// `Cross`.
///
/// A member of a `Monomorphize` family is one of the runtime's values, so the
/// specialized representation needs no split into a run and a value the way
/// `Cross` does: the run is the value, here and at every impl.
///
/// `Cross` is not a supertrait, and that is forced. `Option<T>` crosses at
/// both representations, and its `Cross` impl holds only where `T` crosses
/// uniformly; a supertrait would demand that of the specialized impl too,
/// where the element is specialized and nothing says it is also uniform.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not cross the boundary as one of the runtime's values",
    note = "a slice crosses as the two registers it occupies and is no value of the language: it is an argument and a result, never a field, a container's element, or a parameter taken by reference (RFC-0047 rule 6)."
)]
pub trait OneValue<Rt, Rep = Uniform>: Sized + Send + Sync + 'static
where
    Rt: Runtime,
{
    /// `Self` is the runtime's value under another name, with its layout:
    /// a container of `Self` is a container of values in place.
    const STORED_AS_VALUE: bool = false;

    fn erase(self, rt: &Rt) -> Rt::Value;

    /// # Safety
    /// `value` was erased from `Self` (by the runtime's `erase::<Self>` or
    /// `Self::erase`).
    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self;

    /// # Safety
    /// As `Cross::from_run`.
    unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self {
        // SAFETY: the caller's contract, at one value.
        unsafe { Self::materialize(rt, run[0]) }
    }

    /// As `Cross::into_run`.
    fn into_run(self, rt: &Rt, out: &mut [Rt::Value]) {
        out[0] = self.erase(rt);
    }

    /// # Safety
    /// `reference` names a live storage of `Self`, exclusively for the
    /// duration when `deref_mut`.
    unsafe fn deref<'a>(_rt: &Rt, _reference: &'a Rt::Value) -> &'a Self {
        panic!("{}", NO_STORAGE)
    }

    /// # Safety
    /// As `deref`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a>(_rt: &Rt, _reference: &'a Rt::Value) -> &'a mut Self {
        panic!("{}", NO_STORAGE)
    }
}

/// The `Cross` of a type whose crossing is one value: its run is that value,
/// in both directions and at the return position.
#[macro_export]
macro_rules! cross_one_value {
    ($t:ty, at $rt:ty) => {
        impl $crate::Cross<$rt> for $t {
            type Form = $crate::One;
            type ReturnForm = $crate::One;

            unsafe fn from_run(rt: &$rt, run: &[<$rt as $crate::Runtime>::Value]) -> Self {
                // SAFETY: the caller's contract, at one value.
                unsafe { <Self as $crate::OneValue<$rt>>::from_run(rt, run) }
            }

            fn into_run(self, rt: &$rt, out: &mut [<$rt as $crate::Runtime>::Value]) {
                <Self as $crate::OneValue<$rt>>::into_run(self, rt, out)
            }
        }

        impl $crate::Passed<$rt> for $t {
            type As<'a> = Self;

            fn cross(rt: &$rt, passed: Self) -> <$rt as $crate::Runtime>::Value {
                <Self as $crate::OneValue<$rt>>::erase(passed, rt)
            }

            unsafe fn restore<'a>(rt: &$rt, word: <$rt as $crate::Runtime>::Value) -> Self::As<'a> {
                // SAFETY: the caller's contract, which is `materialize`'s.
                unsafe { <Self as $crate::OneValue<$rt>>::materialize(rt, word) }
            }
        }
    };
    ($t:ty $(, $($g:tt)*)?) => {
        impl<$($($g)*,)? __Rt> $crate::Cross<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
            type Form = $crate::One;
            type ReturnForm = $crate::One;

            unsafe fn from_run(
                rt: &__Rt,
                run: &[<__Rt as $crate::Runtime>::Value],
            ) -> Self {
                // SAFETY: the caller's contract, at one value.
                unsafe { <Self as $crate::OneValue<__Rt>>::from_run(rt, run) }
            }

            fn into_run(self, rt: &__Rt, out: &mut [<__Rt as $crate::Runtime>::Value]) {
                <Self as $crate::OneValue<__Rt>>::into_run(self, rt, out)
            }
        }

        impl<$($($g)*,)? __Rt> $crate::Passed<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
            type As<'a> = Self;

            fn cross(rt: &__Rt, passed: Self) -> <__Rt as $crate::Runtime>::Value {
                <Self as $crate::OneValue<__Rt>>::erase(passed, rt)
            }

            unsafe fn restore<'a>(
                rt: &__Rt,
                word: <__Rt as $crate::Runtime>::Value,
            ) -> Self::As<'a> {
                // SAFETY: the caller's contract, which is `materialize`'s.
                unsafe { <Self as $crate::OneValue<__Rt>>::materialize(rt, word) }
            }
        }
    };
}

/// The message a converted type gives when read through a reference: it
/// has no storage of its own type (RFC-0032).
const NO_STORAGE: &str =
    "a value converted at the boundary has no storage of its own type to read through";

/// A type the runtime stores as itself: its `erase` is `rt.erase::<Self>`,
/// so the runtime reads a value erased from it back as a `Self` in place.
///
/// This is what `OneValue` does not say. `Option<T>::erase` is `rt.some(..)`
/// and a derived struct's is a heap object, so `Runtime::value_as_ref::<T>`
/// — which `Erased::as_ref` calls and `Erased::deref` reads out of the value
/// word — is sound for a `Stored` type and for no other.
pub trait Stored<Rt>: OneValue<Rt>
where
    Rt: Runtime,
{
}

/// A name for the runtime's value with its layout, so a `[Rt::Value]` in
/// storage is read in place as a `[Self]` (`Ref::as_slice`).
///
/// # Safety
/// `Self` is `#[repr(transparent)]` with `Rt::Value` as its one
/// non-zero-sized field.
pub unsafe trait TransparentOver<Rt>: OneValue<Rt>
where
    Rt: Runtime,
{
}

/// The `Value -> Self` step a body takes outside the glue: identity for the
/// runtime's own value, otherwise a recursion that ends in
/// `Runtime::materialize`. There is no impl for a bare scalar: a scalar comes
/// out as `Erased<Rt, T>`, whose `from_value` is the crossing for it like any
/// other.
///
/// This is `OneValue::materialize`'s sibling for the recursion a body drives,
/// and it carries the same contract: the crossing is the door. The impl for
/// the runtime's own value is the identity — there is no Rust type a raw value
/// disagrees with — and the impl for `Vec<E>` materializes the buffer and then
/// takes its own element step.
///
/// # Safety
/// `value` was erased from `Self` at a site the checker matched to this
/// parameter's type: the handler's declared parameter type is what `combine`
/// unified the argument against, so the `erase::<Self>` that made this value
/// and this `from_value` name one Rust type. A debug build restates the fact
/// with `debug_assert_erased_from!`; a release build does not look.
pub unsafe trait FromValue<Rt>: Sized
where
    Rt: Runtime,
{
    /// # Safety
    /// The trait's contract.
    unsafe fn from_value(rt: &Rt, value: Rt::Value) -> Self;
}

/// Whether `value` records the `erase::<T>` a crossing's contract names.
///
/// Only `debug_assert_erased_from!` calls this, so the comparison exists on no
/// release path.
#[doc(hidden)]
pub fn is_erased_from<T, Rt>(rt: &Rt, value: &Rt::Value) -> bool
where
    T: 'static,
    Rt: Runtime,
{
    rt.type_of(value) == Some(TypeId::of::<T>())
}

/// States a crossing's contract where a debug build can afford to read it:
/// `$value` was erased from `$t`. It expands to `debug_assert!` and to nothing
/// else, so no release path compares a `TypeId`.
#[macro_export]
macro_rules! debug_assert_erased_from {
    ($rt:expr, $value:expr, $t:ty) => {
        ::core::debug_assert!(
            $crate::is_erased_from::<$t, _>($rt, $value),
            "expected a value erased from `{}`, found {}",
            ::core::any::type_name::<$t>(),
            $crate::erased_description($rt, $value)
        )
    };
}

/// The phrase `debug_assert_erased_from!` puts after "found".
///
/// Only that macro's message calls this, and only when the contract was
/// already broken.
#[doc(hidden)]
pub fn erased_description<Rt>(rt: &Rt, value: &Rt::Value) -> String
where
    Rt: Runtime,
{
    match (rt.type_of(value), rt.type_name_of(value)) {
        (Some(_), Some(name)) => format!("one erased from `{name}`"),
        (Some(found), None) => format!("a payload of {found:?}"),
        (None, _) => "a value no Rust type was erased into".to_owned(),
    }
}

/// A `Stored` type that lives in the runtime's value word itself, so
/// `Erased<R, T>` derefs to it with no runtime in hand.
pub trait Inline: Copy + Send + Sync + 'static {}

/// Calls `$m! { Name: type, ... }` with every `Inline` type and a name for
/// it. `Inline` is implemented from this list, and a runtime that tags its
/// value word per type builds the tag from the same list.
#[macro_export]
macro_rules! for_each_inline {
    ($m:ident) => {
        $m! {
            I8: i8, I16: i16, I32: i32, I64: i64,
            U8: u8, U16: u16, U32: u32, U64: u64,
            F64: f64, Char: char, Bool: bool, Unit: ()
        }
    };
}

macro_rules! inline {
    ($($name:ident: $t:ty),*) => { $(
        const _: () = assert!(
            std::mem::size_of::<$t>() <= 8
                && std::mem::align_of::<$t>() <= 8
                && !std::mem::needs_drop::<$t>(),
            "an Inline type fits the runtime's value word"
        );
        impl Inline for $t {}
    )* };
}
crate::for_each_inline!(inline);

/// A type stored as itself: the runtime keeps the Rust value and hands it
/// back untouched (RFC-0022).
#[macro_export]
macro_rules! cross_as_stored {
    ($t:ty $(, $($g:tt)*)?) => {
        impl<$($($g)*,)? __Rt> $crate::Stored<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
        }

        impl<$($($g)*,)? __Rt> $crate::Borrowable<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
        }

        impl<$($($g)*,)? __Rt> $crate::BorrowableSpecialized<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
        }

        $crate::cross_one_value!($t $(, $($g)*)?);
        $crate::borrowed_as_self!($t $(, $($g)*)?);
        $crate::cross_whole!($crate::Uniform, $t $(, $($g)*)?);
        $crate::cross_whole!($crate::Specialized, $t $(, $($g)*)?);
    };
}

/// The crossing of one representation as the whole Rust value in one runtime
/// box.
#[doc(hidden)]
#[macro_export]
macro_rules! cross_whole {
    ($rep:ty, $t:ty $(, $($g:tt)*)?) => {
        impl<$($($g)*,)? __Rt> $crate::OneValue<__Rt, $rep> for $t
        where
            __Rt: $crate::Runtime,
        {
            $crate::whole_box!($t, __Rt);
        }
    };
}

/// The four methods of a crossing whose runtime box holds the whole Rust
/// value.
#[doc(hidden)]
#[macro_export]
macro_rules! whole_box {
    ($t:ty, $rt:ident) => {
        fn erase(self, rt: &$rt) -> <$rt as $crate::Runtime>::Value {
            // SAFETY: stored as itself (RFC-0022).
            unsafe { rt.erase::<$t>(self) }
        }

        unsafe fn materialize(rt: &$rt, value: <$rt as $crate::Runtime>::Value) -> Self {
            // SAFETY: the caller's contract, and `erase` is `rt.erase::<$t>`.
            unsafe { rt.materialize::<$t>(value) }
        }

        unsafe fn deref<'a>(rt: &$rt, reference: &'a <$rt as $crate::Runtime>::Value) -> &'a Self {
            // SAFETY: the caller's contract.
            unsafe { rt.deref::<$t>(reference) }
        }

        unsafe fn deref_mut<'a>(
            rt: &$rt,
            reference: &'a <$rt as $crate::Runtime>::Value,
        ) -> &'a mut Self {
            // SAFETY: the caller's contract.
            unsafe { rt.deref_mut::<$t>(reference) }
        }
    };
}

cross_as_stored!(i8);
cross_as_stored!(i16);
cross_as_stored!(i32);
cross_as_stored!(i64);
cross_as_stored!(u8);
cross_as_stored!(u16);
cross_as_stored!(u32);
cross_as_stored!(u64);
cross_as_stored!(f64);
cross_as_stored!(char);
cross_as_stored!(bool);
cross_as_stored!(String);
cross_as_stored!(());

crate::cross_one_value!(Never);

impl<Rep, Rt> OneValue<Rt, Rep> for Never
where
    Rt: Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        match self {}
    }

    unsafe fn materialize(_: &Rt, _: Rt::Value) -> Self {
        panic!("a value of type `!` was materialized")
    }
}

/// Whether a container of `T` is stored as a container of `T`: when `T`
/// is the runtime's value or a `repr(transparent)` name for it; any other
/// element was converted on the way in, so the storage holds values, not
/// `T`s.
pub(crate) fn stored_as_container_of<T, Rt>() -> bool
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    T::STORED_AS_VALUE || TypeId::of::<T>() == TypeId::of::<Rt::Value>()
}

/// There is no cast arm beside this downcast: two instantiations of a
/// `repr(Rust)` type, such as `Option<Value>` and `Option<Erased<..>>`,
/// have no layout the language promises to be the same.
pub(crate) fn storage_as<S, T>(stored: &S) -> Option<&T>
where
    S: 'static,
    T: 'static,
{
    (stored as &dyn Any).downcast_ref::<T>()
}

pub(crate) fn storage_as_mut<S, T>(stored: &mut S) -> Option<&mut T>
where
    S: 'static,
    T: 'static,
{
    (stored as &mut dyn Any).downcast_mut::<T>()
}

crate::cross_one_value!(Option<T>, T: OneValue<__Rt>);

impl<T, Rep, Rt> OneValue<Rt, Rep> for Option<T>
where
    T: OneValue<Rt, Rep>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        match self {
            Some(v) => rt.some(v.erase(rt)),
            None => rt.none(),
        }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        if rt.is_none(&value) {
            return None;
        }
        // SAFETY: the caller's contract, forwarded: `erase` put a `T`'s
        // value under the `some`.
        Some(unsafe { T::materialize(rt, rt.unwrap_some(value)) })
    }
}

crate::cross_one_value!(Result<T, E>, T: OneValue<__Rt>, E: OneValue<__Rt>);

/// The erase half of a `Result`'s crossing. The pair handed to the runtime is
/// a `Result<Owned<Rt>, Owned<Rt>>` because that is the type
/// `acvus-interpreter`'s `Runtime::erase` answers with the language's flat
/// variant; erasing a `Result` of anything else there boxes a Rust value
/// instead (RFC-0038, RFC-0048 §7, RFC-0050 rule 8).
fn erase_result<T, E, Rt>(
    value: Result<T, E>,
    rt: &Rt,
    erase_ok: fn(T, &Rt) -> Rt::Value,
    erase_err: fn(E, &Rt) -> Rt::Value,
) -> Rt::Value
where
    Rt: Runtime,
{
    let inner: Result<Owned<Rt>, Owned<Rt>> = value
        .map(|v| Owned::from_value(erase_ok(v, rt)))
        .map_err(|e| Owned::from_value(erase_err(e, rt)));
    // SAFETY: the language's Result is the runtime's
    // `Result<Owned<Rt>, Owned<Rt>>` (RFC-0038, RFC-0048 §7).
    unsafe { rt.erase::<Result<Owned<Rt>, Owned<Rt>>>(inner) }
}

/// # Safety
/// `value` was erased by `erase_result`, and the two payload steps undo the
/// two it was given.
unsafe fn materialize_result<T, E, Rt>(
    rt: &Rt,
    value: Rt::Value,
    materialize_ok: unsafe fn(&Rt, Rt::Value) -> T,
    materialize_err: unsafe fn(&Rt, Rt::Value) -> E,
) -> Result<T, E>
where
    Rt: Runtime,
{
    // SAFETY: the caller's contract, and `erase_result` boxes a
    // `Result<Owned<Rt>, Owned<Rt>>`.
    let inner = unsafe { rt.materialize::<Result<Owned<Rt>, Owned<Rt>>>(value) };
    // SAFETY: the caller's contract, forwarded: each arm's payload was erased
    // by the step this one undoes.
    unsafe {
        inner
            .map(|v| materialize_ok(rt, v.into_value()))
            .map_err(|e| materialize_err(rt, e.into_value()))
    }
}

impl<T, E, Rep, Rt> OneValue<Rt, Rep> for Result<T, E>
where
    T: OneValue<Rt, Rep>,
    E: OneValue<Rt, Rep>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        erase_result(
            self,
            rt,
            <T as OneValue<Rt, Rep>>::erase,
            <E as OneValue<Rt, Rep>>::erase,
        )
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and the two steps are the inverses of
        // the ones `erase` gave.
        unsafe {
            materialize_result(
                rt,
                value,
                <T as OneValue<Rt, Rep>>::materialize,
                <E as OneValue<Rt, Rep>>::materialize,
            )
        }
    }
}

// A `Result` has no `BorrowableSpecialized` impl, and that is a decision: a
// crossed `Result` is the flat heap variant, so no storage anywhere is shaped
// like Rust's `Result<T, E>` for a reference to name. The `borrowed_result`
// compile-fail case in `acvus-extern-macro` pins that refusal at both
// crossings, the concrete one through `Borrowable` and the monomorphized one
// through the marker; adding either impl makes a case there pass silently.

crate::cross_one_value!(Arr<T, N>, T: OneValue<__Rt>, N: Var<kind::Length>);

impl<T, N, Rep, Rt> OneValue<Rt, Rep> for Arr<T, N>
where
    T: OneValue<Rt, Rep>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let items: Vec<Owned<Rt>> = self
            .0
            .into_iter()
            .map(|v| Owned::from_value(v.erase(rt)))
            .collect();
        // SAFETY: the language's array is `Arr<Owned<Rt>, ()>` (RFC-0022,
        // RFC-0048 §7).
        unsafe { rt.erase::<Arr<Owned<Rt>, ()>>(Arr::new(items)) }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and `erase` boxes an
        // `Arr<Owned<Rt>, ()>`.
        let items = unsafe { rt.materialize::<Arr<Owned<Rt>, ()>>(value) };
        // SAFETY: the caller's contract, forwarded: `erase` erased every
        // element from a `T`.
        Arr::new(
            items
                .0
                .into_iter()
                .map(|v| unsafe { T::materialize(rt, v.into_value()) })
                .collect(),
        )
    }

    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self {
        // SAFETY: the caller's contract, and `erase` boxes an
        // `Arr<Owned<Rt>, ()>`.
        let stored = unsafe { rt.deref::<Arr<Owned<Rt>, ()>>(reference) };
        let Some(items) = storage_as::<_, Vec<T>>(&stored.0) else {
            panic!("{NO_STORAGE}")
        };
        // SAFETY: `Arr<T, N>` is `repr(transparent)` over `Vec<T>`.
        unsafe { &*(items as *const Vec<T> as *const Self) }
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self {
        // SAFETY: the caller's contract, exclusively, and `erase` boxes an
        // `Arr<Owned<Rt>, ()>`.
        let stored = unsafe { rt.deref_mut::<Arr<Owned<Rt>, ()>>(reference) };
        let Some(items) = storage_as_mut::<_, Vec<T>>(&mut stored.0) else {
            panic!("{NO_STORAGE}")
        };
        // SAFETY: `Arr<T, N>` is `repr(transparent)` over `Vec<T>`.
        unsafe { &mut *(items as *mut Vec<T> as *mut Self) }
    }
}

/// The language's array keeps a `Vec` of the runtime's values, which is the
/// storage an `Arr<T, N>` reference reads in place.
impl<T, N, Rt> crate::Borrowable<Rt> for Arr<T, N>
where
    T: OneValue<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
}

/// The elements of a container box, each taken by its own `FromValue`; the
/// buffer itself is reused when the element is the value.
///
/// # Safety
/// Each element of `items` was erased from `E`, which is `FromValue`'s
/// contract element by element: the container the checker matched carries one
/// element type.
unsafe fn elements_from_values<E, Rt>(rt: &Rt, items: Vec<Owned<Rt>>) -> Vec<E>
where
    E: FromValue<Rt> + 'static,
    Rt: Runtime,
{
    if TypeId::of::<E>() == TypeId::of::<Rt::Value>() {
        let mut items = ManuallyDrop::new(items);
        // SAFETY: `E` is `Rt::Value` and `Owned<Rt>` is `repr(transparent)`
        // over it: one element layout, one allocator.
        return unsafe {
            Vec::from_raw_parts(items.as_mut_ptr().cast(), items.len(), items.capacity())
        };
    }
    items
        .into_iter()
        // SAFETY: this function's contract, one element at a time.
        .map(|item| unsafe { E::from_value(rt, item.into_value()) })
        .collect()
}

unsafe impl<E, Rt> FromValue<Rt> for Vec<E>
where
    E: FromValue<Rt> + Send + Sync + 'static,
    Rt: Runtime,
{
    unsafe fn from_value(rt: &Rt, value: Rt::Value) -> Self {
        debug_assert_erased_from!(rt, &value, Vec<Owned<Rt>>);
        // SAFETY: the trait's contract. A language list crosses as a
        // `Vec<Owned<Rt>>` box whose elements were each erased from `E`.
        unsafe { elements_from_values(rt, rt.materialize::<Vec<Owned<Rt>>>(value)) }
    }
}

unsafe impl<E, N, Rt> FromValue<Rt> for Arr<E, N>
where
    E: FromValue<Rt> + Send + Sync + 'static,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    unsafe fn from_value(rt: &Rt, value: Rt::Value) -> Self {
        debug_assert_erased_from!(rt, &value, Arr<Owned<Rt>, ()>);
        // SAFETY: the trait's contract, as `Vec<E>`'s impl states it; a
        // language array crosses as an `Arr<Owned<Rt>, ()>` box.
        let items = unsafe { rt.materialize::<Arr<Owned<Rt>, ()>>(value) };
        // SAFETY: as above, element by element.
        Arr::new(unsafe { elements_from_values(rt, items.0) })
    }
}
