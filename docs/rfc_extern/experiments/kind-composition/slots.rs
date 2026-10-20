//! A checked runtime (rfc_extern references §5, first example): every value
//! is held in a slot the runtime checks, a `Value` is the slot's index, and a
//! reference value is the index marked shared or exclusive. A projection
//! through a reference borrows the slot after checking the mark, and a
//! misuse panics.
//! A view of a text or a run is no slot's: it is its start and length, marked
//! as the view it is, and its release ends nothing. Its `unsafe` blocks are
//! the option and result kinds' calls of their payload's crossing, whose
//! premises pass through from the caller and which the slots check on use,
//! and the opaque and view kinds' reads by layout, which an unsafe kind does
//! unchecked (runtime §2, §3).

use std::any::Any;
use std::cell::RefCell;
use std::future::Future;
use std::mem::ManuallyDrop;
use std::pin::Pin;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::Counted;
use crate::boundary::{Cross, TypeVar};
use crate::contract::{
    EnumLayout, EnumTable, Enums, Functions, ObjectLayout, ObjectTable, Objects, Opaques, Options, Programs, Results, Runtime, Scalar,
    Scalars, Strings, Structures, Tuples, Views,
};
use crate::laid::{Laid, Table};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Value(Form);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// The value a slot holds, or a reference to it, as the mark states.
    Slot { slot: usize, mark: Mark },
    /// A view of a text or a run in memory no slot holds.
    View { at: NonNull<u8>, len: usize, viewed: Viewed },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Viewed {
    Text,
    Run,
    /// A run reached through this view alone.
    RunMut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mark {
    /// The value that owns the slot.
    Owner,
    /// A shared reference to the slot's value.
    Shared,
    /// An exclusive reference to the slot's value.
    Exclusive,
}

type Held = Rc<RefCell<Box<dyn Any>>>;

#[derive(Default)]
struct Store {
    slots: Vec<Option<Held>>,
    free: Vec<usize>,
}

thread_local! {
    static STORE: RefCell<Store> = RefCell::new(Store::default());
}

fn hold(content: Box<dyn Any>) -> Value {
    STORE.with(|store| {
        let mut store = store.borrow_mut();
        let held = Some(Rc::new(RefCell::new(content)));
        let slot = match store.free.pop() {
            Some(slot) => {
                store.slots[slot] = held;
                slot
            }
            None => {
                store.slots.push(held);
                store.slots.len() - 1
            }
        };
        Value(Form::Slot { slot, mark: Mark::Owner })
    })
}

/// The slot a value names, with its mark.
fn slot_of(value: Value) -> (usize, Mark) {
    let Value(Form::Slot { slot, mark }) = value else {
        panic!("a view used where a slot's value is");
    };
    (slot, mark)
}

/// The slot a value names; the store itself is not borrowed while the slot
/// is, so a projection may reach other slots.
fn slot(value: Value) -> Held {
    let (slot, _) = slot_of(value);
    STORE.with(|store| store.borrow().slots[slot].clone().expect("a value used after its release"))
}

/// The slot a reference value names, checked to be one.
fn referent(reference: Value) -> Held {
    assert_ne!(slot_of(reference).1, Mark::Owner, "a value opened without a reference to it");
    slot(reference)
}

/// The slot an exclusive reference value names, checked to be one.
fn referent_mut(reference: Value) -> Held {
    assert_eq!(slot_of(reference).1, Mark::Exclusive, "a value written through a reference that is not exclusive");
    slot(reference)
}

fn unhold(value: Value) -> Box<dyn Any> {
    let (slot, mark) = slot_of(value);
    assert_eq!(mark, Mark::Owner, "a value taken by what does not own it");
    let held = STORE.with(|store| {
        let mut store = store.borrow_mut();
        let held = store.slots[slot].take().expect("a value taken or released twice");
        store.free.push(slot);
        held
    });
    Rc::try_unwrap(held).unwrap_or_else(|_| panic!("a value taken while it is lent")).into_inner()
}

/// Ends the value's owner; a reference value or a view ends nothing.
fn release_value(value: Value) {
    if let Value(Form::Slot { mark: Mark::Owner, .. }) = value {
        drop(unhold(value));
    }
}

/// A shared reference value naming the value `value` is.
fn shared(value: Value) -> Value {
    match value.0 {
        Form::Slot { slot, .. } => Value(Form::Slot { slot, mark: Mark::Shared }),
        // A shared reborrow of an exclusive view is a shared view.
        Form::View { at, len, viewed: Viewed::RunMut } => Value(Form::View { at, len, viewed: Viewed::Run }),
        // A reference to a reference is that reference.
        Form::View { viewed: Viewed::Text | Viewed::Run, .. } => value,
    }
}

/// An exclusive reference value naming the value `value` is.
fn exclusive(value: Value) -> Value {
    match value.0 {
        Form::Slot { slot, mark: Mark::Owner | Mark::Exclusive } => Value(Form::Slot { slot, mark: Mark::Exclusive }),
        // A reference to a reference is that reference: a shared one stays
        // the reference it is.
        Form::Slot { mark: Mark::Shared, .. } | Form::View { .. } => value,
    }
}

fn taken<T>(value: Value) -> T
where
    T: 'static,
{
    *unhold(value).downcast::<T>().unwrap_or_else(|_| panic!("a value taken as a type it was not made as"))
}

fn projected<T, O>(reference: Value, lent: impl FnOnce(&T) -> O) -> O
where
    T: 'static,
{
    let held = referent(reference);
    let content = held.borrow();
    lent(content.downcast_ref::<T>().expect("a value projected as a type it was not made as"))
}

fn projected_mut<T, O>(reference: Value, lent: impl FnOnce(&mut T) -> O) -> O
where
    T: 'static,
{
    let held = referent_mut(reference);
    let mut content = held.borrow_mut();
    lent(content.downcast_mut::<T>().expect("a value projected as a type it was not made as"))
}

/// A tuple's parts, each a value the tuple owns.
struct Parts(Vec<Value>);

impl Drop for Parts {
    fn drop(&mut self) {
        self.0.drain(..).for_each(release_value);
    }
}

/// An option's payload, a value the option owns.
struct OptionHeld(Option<Value>);

impl Drop for OptionHeld {
    fn drop(&mut self) {
        self.0.take().into_iter().for_each(release_value);
    }
}

/// A result's payload, a value the result owns; `None` once taken.
struct ResultHeld(Option<Result<Value, Value>>);

impl Drop for ResultHeld {
    fn drop(&mut self) {
        if let Some(Ok(payload) | Err(payload)) = self.0.take() {
            release_value(payload);
        }
    }
}

/// An opaque value's storage, untyped: read by layout at any instantiation of the
/// type it was made as.
struct OpaqueHeld {
    storage: NonNull<u8>,
    drop: unsafe fn(NonNull<u8>),
}

impl Drop for OpaqueHeld {
    fn drop(&mut self) {
        // SAFETY: `storage` is the box `drop` was made for, ended once, here.
        unsafe { (self.drop)(self.storage) }
    }
}

/// # Safety
/// `storage` is a box of `X` this runtime leaked, given up here.
unsafe fn drop_storage<X>(storage: NonNull<u8>) {
    // SAFETY: the caller's premise.
    drop(unsafe { Box::from_raw(storage.cast::<X>().as_ptr()) });
}

/// The runtime's release, as a declared structure's block calls it on each
/// field.
///
/// # Safety
/// As `Runtime::release`.
unsafe fn release_field(value: Value) {
    release_value(value);
}

pub struct Slots;

pub struct SlotScalars;
pub struct SlotStrings;
pub struct SlotTuples;
pub struct SlotOptions;
pub struct SlotResults;
pub struct SlotOpaques;
pub struct SlotObjects;
pub struct SlotEnums;
pub struct SlotFunctions;
pub struct SlotViews;

/// A function value: the program the runtime runs for it. A call through a
/// shared reference borrows it exclusively while it runs, so a call that
/// reaches it again panics.
struct FunctionHeld(RefCell<Box<dyn FnMut(&[Value]) -> Value>>);

/// A function value called asynchronously: each call gives the future that
/// runs its body.
struct AsyncFunctionHeld(RefCell<Box<dyn FnMut(&[Value]) -> Pin<Box<dyn Future<Output = Value>>>>>);

impl Runtime for Slots {
    type Value = Value;

    type Scalars = SlotScalars;
    type Strings = SlotStrings;
    type Tuples = SlotTuples;
    type Options = SlotOptions;
    type Results = SlotResults;
    type Opaques = SlotOpaques;
    type Objects = SlotObjects;
    type Enums = SlotEnums;
    type Functions = SlotFunctions;
    type Views = SlotViews;

    unsafe fn release(value: Value) {
        release_value(value);
    }

    unsafe fn value_as_ref(place: &Value) -> Value {
        shared(*place)
    }

    unsafe fn value_as_mut(place: &mut Value) -> Value {
        exclusive(*place)
    }
}

impl Counted for Slots {
    fn live() -> usize {
        STORE.with(|store| store.borrow().slots.iter().filter(|slot| slot.is_some()).count())
    }
}

impl Scalars<Slots> for SlotScalars {
    fn give<S>(scalar: S) -> Value
    where
        S: Scalar,
    {
        hold(Box::new(scalar))
    }

    unsafe fn take<S>(value: Value) -> S
    where
        S: Scalar,
    {
        taken(value)
    }

    unsafe fn project<S, O>(reference: Value, lent: impl FnOnce(&S) -> O) -> O
    where
        S: Scalar,
    {
        projected(reference, lent)
    }

    unsafe fn project_mut<S, O>(reference: Value, lent: impl FnOnce(&mut S) -> O) -> O
    where
        S: Scalar,
    {
        projected_mut(reference, lent)
    }
}

impl Strings<Slots> for SlotStrings {
    fn give(string: String) -> Value {
        hold(Box::new(string))
    }

    unsafe fn take(value: Value) -> String {
        taken(value)
    }

    unsafe fn project<O>(reference: Value, lent: impl FnOnce(&String) -> O) -> O {
        projected(reference, lent)
    }

    unsafe fn project_mut<O>(reference: Value, lent: impl FnOnce(&mut String) -> O) -> O {
        projected_mut(reference, lent)
    }
}

impl Tuples<Slots> for SlotTuples {
    unsafe fn give<const N: usize>(parts: [Value; N]) -> Value {
        hold(Box::new(Parts(parts.to_vec())))
    }

    unsafe fn take<const N: usize>(value: Value) -> [Value; N] {
        let mut parts = unhold(value).downcast::<Parts>().unwrap_or_else(|_| panic!("a value taken as a tuple it was not made as"));
        std::mem::take(&mut parts.0).try_into().unwrap_or_else(|_| panic!("a tuple taken at another arity"))
    }

    unsafe fn parts<const N: usize>(reference: Value) -> [Value; N] {
        projected(reference, |parts: &Parts| {
            let parts: &[Value; N] = parts.0.as_slice().try_into().expect("a tuple projected at another arity");
            parts.map(shared)
        })
    }

    unsafe fn parts_mut<const N: usize>(reference: Value) -> [Value; N] {
        projected_mut(reference, |parts: &mut Parts| {
            let parts: &[Value; N] = parts.0.as_slice().try_into().expect("a tuple projected at another arity");
            parts.map(exclusive)
        })
    }
}

impl Options<Slots> for SlotOptions {
    fn give<P>(option: Option<P>) -> Value
    where
        P: Cross<Slots>,
    {
        hold(Box::new(OptionHeld(option.map(Cross::give))))
    }

    unsafe fn take<P>(value: Value) -> Option<P>
    where
        P: Cross<Slots>,
    {
        let mut held: OptionHeld = taken(value);
        // SAFETY: the caller's premise: the payload was given as `P` and is
        // the caller's along with the option.
        held.0.take().map(|payload| unsafe { P::take(payload) })
    }

    unsafe fn project<P, O>(reference: Value, lent: impl FnOnce(&Option<P>) -> O) -> O
    where
        P: TypeVar<Slots>,
    {
        projected(reference, |option: &OptionHeld| {
            // SAFETY: the caller's premise; the projection owns nothing, so it is
            // never dropped.
            let projection = ManuallyDrop::new(option.0.map(|payload| unsafe { <P as TypeVar<Slots>>::take(payload) }));
            lent(&projection)
        })
    }

    unsafe fn project_mut<P, O>(reference: Value, lent: impl FnOnce(&mut Option<P>) -> O) -> O
    where
        P: TypeVar<Slots>,
    {
        projected_mut(reference, |option: &mut OptionHeld| {
            // SAFETY: the caller's premise; the payload moves into the projection
            // and back, so it has one owner throughout.
            let mut projection = option.0.take().map(|payload| unsafe { <P as TypeVar<Slots>>::take(payload) });
            let out = lent(&mut projection);
            option.0 = projection.map(TypeVar::give);
            out
        })
    }
}

impl Results<Slots> for SlotResults {
    fn give<T, E>(result: Result<T, E>) -> Value
    where
        T: Cross<Slots>,
        E: Cross<Slots>,
    {
        hold(Box::new(ResultHeld(Some(result.map(Cross::give).map_err(Cross::give)))))
    }

    unsafe fn take<T, E>(value: Value) -> Result<T, E>
    where
        T: Cross<Slots>,
        E: Cross<Slots>,
    {
        let mut held: ResultHeld = taken(value);
        let payload = held.0.take().expect("a result taken twice");
        // SAFETY: the caller's premise, for either payload.
        payload.map(|ok| unsafe { T::take(ok) }).map_err(|err| unsafe { E::take(err) })
    }

    unsafe fn project<T, E, O>(reference: Value, lent: impl FnOnce(&Result<T, E>) -> O) -> O
    where
        T: TypeVar<Slots>,
        E: TypeVar<Slots>,
    {
        projected(reference, |result: &ResultHeld| {
            let payload = result.0.expect("a result projected after it was taken");
            // SAFETY: as `SlotOptions::project`.
            let projection = ManuallyDrop::new(
                payload
                    .map(|ok| unsafe { <T as TypeVar<Slots>>::take(ok) })
                    .map_err(|err| unsafe { <E as TypeVar<Slots>>::take(err) }),
            );
            lent(&projection)
        })
    }

    unsafe fn project_mut<T, E, O>(reference: Value, lent: impl FnOnce(&mut Result<T, E>) -> O) -> O
    where
        T: TypeVar<Slots>,
        E: TypeVar<Slots>,
    {
        projected_mut(reference, |result: &mut ResultHeld| {
            let payload = result.0.take().expect("a result projected after it was taken");
            // SAFETY: as `SlotOptions::project_mut`.
            let mut projection = payload
                .map(|ok| unsafe { <T as TypeVar<Slots>>::take(ok) })
                .map_err(|err| unsafe { <E as TypeVar<Slots>>::take(err) });
            let out = lent(&mut projection);
            result.0 = Some(projection.map(TypeVar::give).map_err(TypeVar::give));
            out
        })
    }
}

impl Opaques<Slots> for SlotOpaques {
    fn give<X>(value: X) -> Value {
        let storage = NonNull::from(Box::leak(Box::new(value))).cast();
        hold(Box::new(OpaqueHeld { storage, drop: drop_storage::<X> }))
    }

    unsafe fn take<X>(value: Value) -> X {
        let held = ManuallyDrop::new(taken::<OpaqueHeld>(value));
        // SAFETY: the caller's premise: made as `X`, or as an instantiation of its type of
        // the same layout (variables §4); the storage is given up here.
        *unsafe { Box::from_raw(held.storage.cast::<X>().as_ptr()) }
    }

    unsafe fn project<X, O>(reference: Value, lent: impl FnOnce(&X) -> O) -> O {
        // SAFETY: the caller's premise; the slot's borrow keeps the storage
        // from being written while `lent` runs.
        projected(reference, |held: &OpaqueHeld| lent(unsafe { held.storage.cast::<X>().as_ref() }))
    }

    unsafe fn project_mut<X, O>(reference: Value, lent: impl FnOnce(&mut X) -> O) -> O {
        // SAFETY: as `project`, exclusive.
        projected_mut(reference, |held: &mut OpaqueHeld| lent(unsafe { held.storage.cast::<X>().as_mut() }))
    }
}

impl Objects<Slots> for SlotObjects {
    fn give<X>(value: X) -> Value
    where
        X: ObjectLayout,
    {
        // SAFETY: `X::TABLE` states `X`'s layout (`ObjectLayout`'s premise).
        hold(Box::new(unsafe { Laid::put(value, Table::Object(X::TABLE), release_field) }))
    }

    unsafe fn take<X>(value: Value) -> X
    where
        X: ObjectLayout,
    {
        // SAFETY: the caller's premise.
        unsafe { taken::<Laid<Value>>(value).take() }
    }

    unsafe fn project<X, O>(reference: Value, lent: impl FnOnce(&X) -> O) -> O
    where
        X: ObjectLayout,
    {
        // SAFETY: the caller's premise; the slot's borrow keeps the block
        // from being written while `lent` runs.
        projected(reference, |laid: &Laid<Value>| lent(unsafe { laid.project() }))
    }

    unsafe fn project_mut<X, O>(reference: Value, lent: impl FnOnce(&mut X) -> O) -> O
    where
        X: ObjectLayout,
    {
        // SAFETY: as `project`, exclusive.
        projected_mut(reference, |laid: &mut Laid<Value>| lent(unsafe { laid.project_mut() }))
    }
}

impl Enums<Slots> for SlotEnums {
    fn give<X>(value: X) -> Value
    where
        X: EnumLayout,
    {
        // SAFETY: `X::TABLE` states `X`'s layout (`EnumLayout`'s premise).
        hold(Box::new(unsafe { Laid::put(value, Table::Enum(X::TABLE), release_field) }))
    }

    unsafe fn take<X>(value: Value) -> X
    where
        X: EnumLayout,
    {
        // SAFETY: the caller's premise.
        unsafe { taken::<Laid<Value>>(value).take() }
    }

    unsafe fn project<X, O>(reference: Value, lent: impl FnOnce(&X) -> O) -> O
    where
        X: EnumLayout,
    {
        // SAFETY: as `SlotObjects::project`.
        projected(reference, |laid: &Laid<Value>| lent(unsafe { laid.project() }))
    }

    unsafe fn project_mut<X, O>(reference: Value, lent: impl FnOnce(&mut X) -> O) -> O
    where
        X: EnumLayout,
    {
        // SAFETY: as `SlotObjects::project_mut`.
        projected_mut(reference, |laid: &mut Laid<Value>| lent(unsafe { laid.project_mut() }))
    }
}

impl Structures for Slots {
    unsafe fn field(reference: Value, name: &str) -> Value {
        let place = projected(reference, |laid: &Laid<Value>| laid.place(name));
        // SAFETY: the place holds the field's value, which the structure owns
        // and keeps while the reference lives.
        let field = unsafe { place.read() };
        match slot_of(reference).1 {
            Mark::Exclusive => exclusive(field),
            Mark::Shared => shared(field),
            Mark::Owner => unreachable!("a field reached through no reference, which `projected` refuses"),
        }
    }

    unsafe fn variant(reference: Value) -> &'static str {
        projected(reference, |laid: &Laid<Value>| laid.variant())
    }

    unsafe fn make_object(table: &'static ObjectTable, fields: &[(&str, Value)]) -> Value {
        // SAFETY: the caller's premise.
        hold(Box::new(unsafe { Laid::make_object(table, fields, release_field) }))
    }

    unsafe fn make_enum(table: &'static EnumTable, variant: &str, fields: &[(&str, Value)]) -> Value {
        // SAFETY: the caller's premise.
        hold(Box::new(unsafe { Laid::make_enum(table, variant, fields, release_field) }))
    }
}

impl Functions<Slots> for SlotFunctions {
    unsafe fn call_ref<const N: usize>(function: Value, arguments: [Value; N]) -> Value {
        projected(function, |held: &FunctionHeld| (held.0.borrow_mut())(&arguments))
    }

    unsafe fn call_mut<const N: usize>(function: Value, arguments: [Value; N]) -> Value {
        projected_mut(function, |held: &mut FunctionHeld| (held.0.get_mut())(&arguments))
    }

    unsafe fn call_once<const N: usize>(function: Value, arguments: [Value; N]) -> Value {
        let held: FunctionHeld = taken(function);
        (held.0.into_inner())(&arguments)
    }

    unsafe fn call_async_ref<const N: usize>(function: Value, arguments: [Value; N]) -> impl Future<Output = Value> {
        projected(function, |held: &AsyncFunctionHeld| (held.0.borrow_mut())(&arguments))
    }

    unsafe fn call_async_mut<const N: usize>(function: Value, arguments: [Value; N]) -> impl Future<Output = Value> {
        projected_mut(function, |held: &mut AsyncFunctionHeld| (held.0.get_mut())(&arguments))
    }

    unsafe fn call_async_once<const N: usize>(function: Value, arguments: [Value; N]) -> impl Future<Output = Value> {
        let held: AsyncFunctionHeld = taken(function);
        let call = (held.0.borrow_mut())(&arguments);
        // The call owns its function until it ends: a future dropped between
        // its polls drops the function with it.
        async move {
            let out = call.await;
            drop(held);
            out
        }
    }
}

impl Programs for Slots {
    fn function(program: impl FnMut(&[Value]) -> Value + 'static) -> Value {
        hold(Box::new(FunctionHeld(RefCell::new(Box::new(program)))))
    }

    fn async_function(program: impl FnMut(&[Value]) -> Pin<Box<dyn Future<Output = Value>>> + 'static) -> Value {
        hold(Box::new(AsyncFunctionHeld(RefCell::new(Box::new(program)))))
    }
}

impl Views<Slots> for SlotViews {
    fn give_text(text: &str) -> Value {
        Value(Form::View { at: NonNull::from(text).cast(), len: text.len(), viewed: Viewed::Text })
    }

    fn give_run<T>(run: &[T]) -> Value
    where
        T: TypeVar<Slots>,
    {
        Value(Form::View { at: NonNull::from(run).cast(), len: run.len(), viewed: Viewed::Run })
    }

    fn give_run_mut<T>(run: &mut [T]) -> Value
    where
        T: TypeVar<Slots>,
    {
        let len = run.len();
        Value(Form::View { at: NonNull::from(run).cast(), len, viewed: Viewed::RunMut })
    }

    unsafe fn take_text<'a>(view: Value) -> &'a str {
        let Value(Form::View { at, len, viewed: Viewed::Text }) = view else {
            panic!("a text taken from what is no text view");
        };
        // SAFETY: the caller's premise: the view's start and length are a
        // live text's.
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(at.as_ptr(), len)) }
    }

    unsafe fn take_run<'a, T>(view: Value) -> &'a [T]
    where
        T: TypeVar<Slots>,
    {
        let Value(Form::View { at, len, viewed: Viewed::Run | Viewed::RunMut }) = view else {
            panic!("a run taken from what is no run view");
        };
        // SAFETY: the caller's premise: the view's start and length are a
        // live run's of `T`'s layout.
        unsafe { std::slice::from_raw_parts(at.cast::<T>().as_ptr(), len) }
    }

    unsafe fn take_run_mut<'a, T>(view: Value) -> &'a mut [T]
    where
        T: TypeVar<Slots>,
    {
        let Value(Form::View { at, len, viewed: Viewed::RunMut }) = view else {
            panic!("a run written through what is no exclusive run view");
        };
        // SAFETY: as `take_run`, the run reached through this view alone.
        unsafe { std::slice::from_raw_parts_mut(at.cast::<T>().as_ptr(), len) }
    }
}
