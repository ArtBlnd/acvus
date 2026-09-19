//! RFC-0048 rule 7 at its contract: a Rust holder releases the runtime
//! value it owns, exactly once.
//!
//! The RFC's own "Rejected" section records why a test is the only form
//! this contract has: `Value: Copy` does not break the build, so a holder
//! whose field is written back as `Rt::Value` compiles and leaks. Each
//! test here states the contract as a count — put a value whose payload
//! counts its own drops into the holder, let the holder go, and the count
//! is exactly 1; where the holder gives the value back, the count is 0
//! while it is out and 1 when its receiver drops it.

use std::any::{Any, TypeId, type_name};
use std::future::Ready;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{
    Arr, Astr, CallToken, Erased, Fn1, FromValue, FxHashMap, Interner, OneValue, Opaque, Owned,
    Ref, Release, Runtime, cross_as_stored, materialize_field, materialize_payload,
};

// -- A payload that counts its own drops --------------------------------

#[derive(Clone, Default)]
struct Drops(Arc<AtomicUsize>);

impl Drops {
    fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    fn payload(&self) -> Tracked {
        Tracked(self.0.clone())
    }
}

struct Tracked(Arc<AtomicUsize>);

impl Drop for Tracked {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

cross_as_stored!(Tracked);

// -- A runtime whose value is `Copy` ------------------------------------

/// `Release: Copy` forbids a value that owns its payload inline, so this
/// fixture's value is one word naming a cell that `release` frees.
#[derive(Clone, Copy, Debug, Default)]
enum V {
    #[default]
    None,
    Some(*mut V),
    Boxed(*mut (dyn Any + Send + Sync)),
    Reference(*const V),
}

// SAFETY: a cell is reached only through the value that owns it, and a
// `Reference` only while its target is live (RFC-0018).
unsafe impl Send for V {}
unsafe impl Sync for V {}

impl Release for V {
    fn release(self) {
        match self {
            V::None | V::Reference(_) => {}
            // SAFETY: `some` leaked this cell and nothing else releases it.
            V::Some(cell) => unsafe { *Box::from_raw(cell) }.release(),
            // SAFETY: `erase` leaked this cell and nothing else frees it.
            V::Boxed(cell) => drop(unsafe { Box::from_raw(cell) }),
        }
    }
}

#[derive(Clone, Default)]
struct Counted;

/// This runtime's registry declares no sliceable container, so the pair a
/// slice would occupy is never built or read.
const NO_SLICES: &str = "this runtime holds no slices";

static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

/// The shape `Fn1` and the `Iter` stages carry as a closure value.
type UnaryClosure = Box<dyn Fn(&Counted, V) -> V + Send + Sync>;

fn cell_ref(value: &V) -> &(dyn Any + Send + Sync) {
    let V::Boxed(cell) = value else {
        panic!("not an erased value: {value:?}")
    };
    // SAFETY: the value is live, so its cell is.
    unsafe { &**cell }
}

fn open_ref<T>(value: &V) -> &T
where
    T: Send + Sync + 'static,
{
    cell_ref(value)
        .downcast_ref::<T>()
        .unwrap_or_else(|| panic!("value is not a {}", type_name::<T>()))
}

fn open_mut<T>(value: &mut V) -> &mut T
where
    T: Send + Sync + 'static,
{
    let V::Boxed(cell) = value else {
        panic!("not an erased value: {value:?}")
    };
    // SAFETY: `&mut V` is the exclusive name of the value and its cell.
    unsafe { &mut **cell }
        .downcast_mut::<T>()
        .unwrap_or_else(|| panic!("value is not a {}", type_name::<T>()))
}

acvus_extern::cross_one_value!(V, at Counted);

impl acvus_extern::OneValue<Counted> for V {
    fn erase(self, _: &Counted) -> V {
        self
    }

    unsafe fn materialize(_: &Counted, value: V) -> Self {
        value
    }

    unsafe fn deref<'a>(_: &Counted, reference: &'a V) -> &'a V {
        let V::Reference(target) = reference else {
            panic!("not a reference: {reference:?}")
        };
        // SAFETY: the target is live for as long as the reference is used.
        unsafe { &**target }
    }

    unsafe fn deref_mut<'a>(_: &Counted, reference: &'a V) -> &'a mut V {
        let V::Reference(target) = reference else {
            panic!("not a reference: {reference:?}")
        };
        // SAFETY: the target is live and, by the checker, exclusively named.
        unsafe { &mut *(*target as *mut V) }
    }
}

impl FromValue<Counted> for V {
    fn from_value(_: &Counted, value: V) -> V {
        value
    }
}

impl Runtime for Counted {
    type Op = acvus_extern::DirectOp<Counted>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = acvus_extern::DirectOp<Counted>;
    type FusedShape = ();

    acvus_extern::direct_call_forms!();

    type Value = V;
    type Frame<'a> = ();
    type Rooted = ();
    type CallFuture<'a> = Ready<V>;

    fn rooted(&self) {}
    fn frame_of(_: &mut ()) {}

    fn type_of(&self, value: &V) -> Option<TypeId> {
        match value {
            V::Boxed(_) => Some(cell_ref(value).type_id()),
            V::None | V::Some(_) | V::Reference(_) => None,
        }
    }

    fn type_name_of(&self, _: &V) -> Option<&'static str> {
        None
    }

    unsafe fn materialize<T>(&self, value: V) -> T
    where
        T: Send + Sync + 'static,
    {
        let V::Boxed(cell) = value else {
            panic!("not an erased value: {value:?}")
        };
        // SAFETY: the caller's contract; the cell is taken, not released.
        let boxed = unsafe { Box::from_raw(cell) };
        *boxed
            .downcast::<T>()
            .unwrap_or_else(|_| panic!("value is not a {}", type_name::<T>()))
    }

    unsafe fn erase<T>(&self, value: T) -> V
    where
        T: Send + Sync + 'static,
    {
        let boxed: Box<dyn Any + Send + Sync> = Box::new(value);
        V::Boxed(Box::into_raw(boxed))
    }

    unsafe fn value_as_ref<'a, T>(&'a self, value: &'a V) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        open_ref::<T>(value)
    }

    unsafe fn value_as_mut<'a, T>(&'a self, value: &'a mut V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        open_mut::<T>(value)
    }

    unsafe fn inline_ref<T>(value: &V) -> &T
    where
        T: acvus_extern::Inline,
    {
        open_ref::<T>(value)
    }

    unsafe fn inline_mut<T>(value: &mut V) -> &mut T
    where
        T: acvus_extern::Inline,
    {
        open_mut::<T>(value)
    }

    unsafe fn deref<'a, T>(&self, reference: &'a V) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        let target = unsafe { <V as OneValue<Counted>>::deref(self, reference) };
        open_ref::<T>(target)
    }

    unsafe fn deref_mut<'a, T>(&self, reference: &'a V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract, exclusively.
        let target = unsafe { <V as OneValue<Counted>>::deref_mut(self, reference) };
        open_mut::<T>(target)
    }

    fn none(&self) -> V {
        V::None
    }

    fn some(&self, payload: V) -> V {
        V::Some(Box::into_raw(Box::new(payload)))
    }

    fn is_none(&self, value: &V) -> bool {
        matches!(value, V::None)
    }

    fn unwrap_some(&self, value: V) -> V {
        let V::Some(cell) = value else {
            panic!("not a Some: {value:?}")
        };
        // SAFETY: `some` leaked this cell; the payload moves out of it.
        *unsafe { Box::from_raw(cell) }
    }

    fn symbol(&self, name: &str) -> Astr {
        SYMBOLS.intern(name)
    }

    fn slice_into_run(&self, _: acvus_extern::Words, _: &mut [V]) {
        panic!("{NO_SLICES}")
    }

    unsafe fn slice_from_run(&self, _: &[V]) -> acvus_extern::Words {
        panic!("{NO_SLICES}")
    }

    unsafe fn reference(&self, target: &V) -> V {
        V::Reference(target as *const V)
    }

    fn call_is_sync(&self, _: &V) -> bool {
        true
    }

    fn call_now<A>(&self, f: &V, _: &mut (), args: A, _: CallToken) -> V
    where
        A: acvus_extern::IntoRun<Self>,
    {
        let run = run_of(self, args);
        let [a] = run.as_slice() else {
            panic!("Counted runs only unary closures")
        };
        open_ref::<UnaryClosure>(f)(self, *a)
    }

    fn call_0<'a>(&'a self, _: &'a V, _: CallToken) -> Ready<V> {
        panic!("Counted runs only unary closures")
    }

    fn call_1<'a>(&'a self, f: &'a V, a: V, _: CallToken) -> Ready<V> {
        std::future::ready(open_ref::<UnaryClosure>(f)(self, a))
    }

    fn call_n<'a>(&'a self, _: &'a V, _: &mut [V], _: CallToken) -> Ready<V> {
        panic!("Counted runs only unary closures")
    }
}

fn tracked_value(rt: &Counted, drops: &Drops) -> V {
    // SAFETY: `cross_as_stored!` makes `Tracked` stored as itself.
    unsafe { rt.erase::<Tracked>(drops.payload()) }
}

fn closure_owning_a_tracked_capture(rt: &Counted, drops: &Drops) -> V {
    let captured = drops.payload();
    let f: UnaryClosure = Box::new(move |rt, argument| {
        argument.release();
        // SAFETY: `usize` is stored as itself.
        unsafe { rt.erase::<usize>(captured.0.load(Ordering::SeqCst)) }
    });
    // SAFETY: a closure is stored as itself.
    unsafe { rt.erase::<UnaryClosure>(f) }
}

// -- `Owned<R>`: the holder every other holder is made of ----------------

#[test]
fn an_owned_releases_its_value_once() {
    let rt = Counted;
    let drops = Drops::default();
    {
        let _held = Owned::<Counted>::from_value(tracked_value(&rt, &drops));
        assert_eq!(drops.count(), 0, "the holder has not been let go of yet");
    }
    assert_eq!(drops.count(), 1, "the holder released its value once");
}

#[test]
fn an_owned_that_gave_its_value_back_releases_nothing() {
    let rt = Counted;
    let drops = Drops::default();
    let value = Owned::<Counted>::from_value(tracked_value(&rt, &drops)).into_value();
    assert_eq!(drops.count(), 0, "the value is out of the holder");
    value.release();
    assert_eq!(drops.count(), 1, "its new owner released it once");
}

// -- `Erased<R, T>` ------------------------------------------------------

#[test]
fn an_erased_releases_its_value_once() {
    let rt = Counted;
    let drops = Drops::default();
    {
        let _held = Erased::<Counted, Tracked>::new(&rt, drops.payload());
        assert_eq!(drops.count(), 0, "the holder has not been let go of yet");
    }
    assert_eq!(drops.count(), 1, "the erased released its value once");
}

#[test]
fn an_erased_that_gave_its_inner_back_is_released_by_its_receiver() {
    let rt = Counted;
    let drops = Drops::default();
    let inner = Erased::<Counted, Tracked>::new(&rt, drops.payload()).into_inner(&rt);
    assert_eq!(drops.count(), 0, "the payload is out of the holder");
    drop(inner);
    assert_eq!(drops.count(), 1, "its receiver dropped it once");
}

#[test]
fn an_erased_that_gave_its_value_back_releases_nothing() {
    let rt = Counted;
    let drops = Drops::default();
    let value = Erased::<Counted, Tracked>::new(&rt, drops.payload()).into_value();
    assert_eq!(drops.count(), 0, "the value is out of the holder");
    value.release();
    assert_eq!(drops.count(), 1, "its new owner released it once");
}

// -- The `Fn*` carriers --------------------------------------------------

#[test]
fn a_closure_carrier_releases_its_closure_once() {
    let rt = Counted;
    let drops = Drops::default();
    {
        let _held =
            Fn1::<V, V, Opaque, Counted>::new(&rt, closure_owning_a_tracked_capture(&rt, &drops));
        assert_eq!(drops.count(), 0, "the holder has not been let go of yet");
    }
    assert_eq!(drops.count(), 1, "the carrier released its closure once");
}

#[test]
fn a_closure_carrier_that_gave_its_value_back_releases_nothing() {
    let rt = Counted;
    let drops = Drops::default();
    let value =
        Fn1::<V, V, Opaque, Counted>::new(&rt, closure_owning_a_tracked_capture(&rt, &drops))
            .into_value();
    assert_eq!(drops.count(), 0, "the closure is out of the carrier");
    value.release();
    assert_eq!(drops.count(), 1, "its new owner released it once");
}

// -- `Vec<Owned<Rt>>`: a container's elements ----------------------------

#[test]
fn a_vec_releases_its_elements_once() {
    let rt = Counted;
    let drops = Drops::default();
    let stored = vec![drops.payload(), drops.payload(), drops.payload()].erase(&rt);
    assert_eq!(drops.count(), 0, "the elements are in the store");
    stored.release();
    assert_eq!(drops.count(), 3, "each element was released once");
}

#[test]
fn a_vec_materialized_back_is_released_by_its_receiver() {
    let rt = Counted;
    let drops = Drops::default();
    let stored = vec![drops.payload(), drops.payload()].erase(&rt);
    // SAFETY: `stored` was erased from this same `Vec<Tracked>`.
    let items = unsafe { <Vec<Tracked> as OneValue<Counted>>::materialize(&rt, stored) };
    assert_eq!(drops.count(), 0, "the elements are out of the store");
    drop(items);
    assert_eq!(drops.count(), 2, "the receiver dropped each element once");
}

// -- `Arr<Owned<Rt>, ()>`: the language's array --------------------------

#[test]
fn an_array_releases_its_elements_once() {
    let rt = Counted;
    let drops = Drops::default();
    let stored = Arr::<Tracked, ()>::new(vec![drops.payload(), drops.payload()]).erase(&rt);
    assert_eq!(drops.count(), 0, "the elements are in the store");
    stored.release();
    assert_eq!(drops.count(), 2, "each element was released once");
}

#[test]
fn an_array_materialized_back_is_released_by_its_receiver() {
    let rt = Counted;
    let drops = Drops::default();
    let stored = Arr::<Tracked, ()>::new(vec![drops.payload(), drops.payload()]).erase(&rt);
    // SAFETY: `stored` was erased from this same `Arr<Tracked, ()>`.
    let items = unsafe { <Arr<Tracked, ()> as OneValue<Counted>>::materialize(&rt, stored) };
    assert_eq!(drops.count(), 0, "the elements are out of the store");
    drop(items);
    assert_eq!(drops.count(), 2, "the receiver dropped each element once");
}

// -- `Result<Owned<Rt>, Owned<Rt>>` --------------------------------------

#[test]
fn a_result_releases_its_ok_payload_once() {
    let rt = Counted;
    let drops = Drops::default();
    let stored = Ok::<Tracked, Tracked>(drops.payload()).erase(&rt);
    assert_eq!(drops.count(), 0, "the payload is in the store");
    stored.release();
    assert_eq!(drops.count(), 1, "the ok payload was released once");
}

#[test]
fn a_result_releases_its_err_payload_once() {
    let rt = Counted;
    let drops = Drops::default();
    let stored = Err::<Tracked, Tracked>(drops.payload()).erase(&rt);
    assert_eq!(drops.count(), 0, "the payload is in the store");
    stored.release();
    assert_eq!(drops.count(), 1, "the err payload was released once");
}

// -- `Option`: the language's option (RFC-0022) --------------------------

#[test]
fn an_option_releases_its_payload_once() {
    let rt = Counted;
    let drops = Drops::default();
    let stored = Some(drops.payload()).erase(&rt);
    assert_eq!(drops.count(), 0, "the payload is in the store");
    stored.release();
    assert_eq!(drops.count(), 1, "the payload was released once");
}

// -- `Obj<Owned<Rt>>`: an object's fields --------------------------------

fn one_field(rt: &Counted, drops: &Drops) -> FxHashMap<Astr, Owned<Counted>> {
    let mut fields = FxHashMap::default();
    fields.insert(
        rt.symbol("payload"),
        Owned::from_value(tracked_value(rt, drops)),
    );
    fields
}

#[test]
fn an_objects_fields_are_released_once() {
    let rt = Counted;
    let drops = Drops::default();
    {
        let _fields = one_field(&rt, &drops);
        assert_eq!(drops.count(), 0, "the holder has not been let go of yet");
    }
    assert_eq!(drops.count(), 1, "the field was released once");
}

#[test]
fn an_object_field_taken_out_is_released_by_its_receiver() {
    let rt = Counted;
    let drops = Drops::default();
    let mut fields = one_field(&rt, &drops);
    // SAFETY: the field was erased from a `Tracked`.
    let taken = unsafe { materialize_field::<Tracked, Counted>(&rt, &mut fields, "payload") };
    drop(fields);
    assert_eq!(drops.count(), 0, "the field is out of the object");
    drop(taken);
    assert_eq!(drops.count(), 1, "its receiver dropped it once");
}

// -- `Variant<Owned<Rt>>`: a variant's payload ---------------------------

fn one_payload(rt: &Counted, drops: &Drops) -> Option<Owned<Counted>> {
    Some(Owned::from_value(tracked_value(rt, drops)))
}

#[test]
fn a_variant_payload_is_released_once() {
    let rt = Counted;
    let drops = Drops::default();
    {
        let _payload = one_payload(&rt, &drops);
        assert_eq!(drops.count(), 0, "the holder has not been let go of yet");
    }
    assert_eq!(drops.count(), 1, "the payload was released once");
}

#[test]
fn a_variant_payload_taken_out_is_released_by_its_receiver() {
    let rt = Counted;
    let drops = Drops::default();
    let payload = one_payload(&rt, &drops);
    // SAFETY: the payload was erased from a `Tracked`.
    let taken = unsafe { materialize_payload::<Tracked, Counted>(&rt, payload, "Tag") };
    assert_eq!(drops.count(), 0, "the payload is out of the variant");
    drop(taken);
    assert_eq!(drops.count(), 1, "its receiver dropped it once");
}

// -- `Ref`: the borrowed path releases nothing ---------------------------

#[test]
fn a_borrow_releases_nothing_and_its_storage_still_releases_once() {
    let rt = Counted;
    let drops = Drops::default();
    let storage = vec![drops.payload(), drops.payload()].erase(&rt);
    {
        let lent = Ref::<Vec<Tracked>, Counted>::lend(&rt, &storage);
        assert_eq!(
            lent.elements(&rt).len(),
            2,
            "the borrow reads the storage in place"
        );
    }
    assert_eq!(
        drops.count(),
        0,
        "a borrow owns nothing and releases nothing"
    );
    storage.release();
    assert_eq!(drops.count(), 2, "the storage released each element once");
}

/// The arguments of a closure call, read back as the run this runtime's own
/// `call_now` reads them from.
fn run_of<Rt, A>(rt: &Rt, args: A) -> Vec<Rt::Value>
where
    Rt: acvus_extern::Runtime,
    A: acvus_extern::IntoRun<Rt>,
{
    let mut run = vec![Rt::Value::default(); A::WIDTH];
    args.into_run(rt, &mut run);
    run
}
