//! RFC-0048 rule 7 at the extension crate's own holders: a `Deque` and an
//! `Items` source release what they own, exactly once.
//!
//! The adaptor stages are not here. A `Map` or a `Filter` holds the
//! `iter::next` instance of the stage below it, and an instance is minted
//! only by a call site the checker resolved, so no adaptor can be built
//! from Rust and no test outside a script can drain one.
//!
//! The runtime fixture is the one `acvus-extern/tests/owned_holders.rs`
//! uses, repeated here because a test target cannot import another
//! crate's test target.

use std::any::{Any, type_name};
use std::future::Ready;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_ext::{Deque, Items};
use acvus_extern::{
    Astr, Canonical, Interner, Owned, Release, Runtime, Var, cross_as_stored, kind,
};

/// No registry these tests combine declares a sliceable container, so the
/// pair a slice would occupy is never built or read.
const NO_SLICES: &str = "this runtime holds no slices";

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

impl Var<kind::Type> for Tracked {}

// SAFETY: `Tracked` holds no `Erased`.
unsafe impl Canonical<kind::Type> for Tracked {
    type Canon = Self;
}

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

/// This runtime's registry declares no enum, so no value of it is ever a
/// variant and no tag register is ever written or read.
const NO_VARIANTS: &str = "this runtime holds no variants";

static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

/// The shape `Closure` and the `Iter` stages carry as a closure value.
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
    fn erase(self, _: acvus_extern::Crossing<'_, Counted>) -> V {
        self
    }

    unsafe fn materialize(_: acvus_extern::Crossing<'_, Counted>, value: V) -> Self {
        value
    }
}

impl acvus_extern::Borrowable<Counted> for V {
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


/// This test plays the runtime, which holds a bare value in an `Owned`.
fn runtime_holding() -> acvus_extern::Holding<'static, Counted> {
    // SAFETY: the test is the runtime, and each word it holds was made for
    // the holder it moves into.
    unsafe { acvus_extern::Holding::new() }
}

impl Runtime for Counted {
    fn instance_value(_: &acvus_extern::InstanceEntry<Self>) -> Self::Value {
        panic!("Counted declares no instances")
    }

    unsafe fn instance_entry<'a>(_: &'a Self::Value) -> &'a acvus_extern::InstanceEntry<Self> {
        panic!("Counted declares no instances")
    }

    type Op = acvus_extern::DirectOp<Counted>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = acvus_extern::DirectOp<Counted>;
    type FusedShape = ();

    acvus_extern::direct_call_forms!();

    type Value = V;
    type Frame<'a> = ();
    type Rooted<'a> = acvus_extern::Ctx<'a, Self>;
    type CallFuture<'a> = Ready<V>;

    fn rooted(&self) -> acvus_extern::Ctx<'_, Self> {
        // SAFETY: the frame is `()`, which names no cells.
        unsafe { acvus_extern::Ctx::new(self, ()) }
    }
    unsafe fn ctx_of<'a, 'r>(
        rooted: &'r mut acvus_extern::Ctx<'a, Self>,
    ) -> &'r mut acvus_extern::Ctx<'a, Self>
    where
        'a: 'r,
    {
        rooted
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
        let target = unsafe { <V as acvus_extern::Borrowable<Counted>>::deref(self, reference) };
        open_ref::<T>(target)
    }

    unsafe fn deref_mut<'a, T>(&self, reference: &'a V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract, exclusively.
        let target =
            unsafe { <V as acvus_extern::Borrowable<Counted>>::deref_mut(self, reference) };
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

    unsafe fn some_at<'a>(&self, value: &'a V) -> Option<&'a V> {
        let V::Some(cell) = value else {
            return None;
        };
        // SAFETY: `some` leaked this cell, and it lives as long as the
        // option that names it.
        Some(unsafe { &**cell })
    }

    unsafe fn some_at_mut<'a>(&self, value: &'a mut V) -> Option<&'a mut V> {
        let V::Some(cell) = value else {
            return None;
        };
        // SAFETY: as `some_at`, with the caller's exclusive loan.
        Some(unsafe { &mut **cell })
    }

    fn symbol(&self, name: &str) -> Astr {
        SYMBOLS.intern(name)
    }

    fn variant_tag(&self, _: &str) -> Self::Value {
        panic!("{NO_VARIANTS}")
    }

    unsafe fn tag_symbol(&self, _: &Self::Value) -> Astr {
        panic!("{NO_VARIANTS}")
    }

    fn undef(&self) -> Self::Value {
        panic!("{NO_VARIANTS}")
    }

    fn is_undef(&self, _: &Self::Value) -> bool {
        panic!("{NO_VARIANTS}")
    }

    fn slice_into_run(&self, _: acvus_extern::Words, _: &mut [Self::Value]) {
        panic!("{NO_SLICES}")
    }

    unsafe fn slice_from_run(&self, _: &[Self::Value]) -> acvus_extern::Words {
        panic!("{NO_SLICES}")
    }

    unsafe fn reference(&self, target: &V) -> V {
        V::Reference(target as *const V)
    }
    fn sleep(
        &self,
        d: std::time::Duration,
    ) -> impl std::future::Future<Output = ()> + Send + use<> {
        async move { std::thread::sleep(d) }
    }

    fn call_is_sync(&self, _: &V) -> bool {
        true
    }

    unsafe fn call_now<A>(&self, f: &V, _: &mut acvus_extern::Ctx<'_, Self>, args: A) -> V
    where
        A: acvus_extern::IntoRun<Self>,
    {
        let run = run_of(self, args);
        let [a] = run.as_slice() else {
            panic!("Counted runs only unary closures")
        };
        open_ref::<UnaryClosure>(f)(self, *a)
    }

    unsafe fn call_0<'a>(&'a self, _: &'a V) -> Ready<V> {
        panic!("Counted runs only unary closures")
    }

    unsafe fn call_1<'a>(&'a self, f: &'a V, a: V) -> Ready<V> {
        std::future::ready(open_ref::<UnaryClosure>(f)(self, a))
    }

    unsafe fn call_n<'a>(&'a self, _: &'a V, _: &mut [V]) -> Ready<V> {
        panic!("Counted runs only unary closures")
    }
}

fn tracked_value(rt: &Counted, drops: &Drops) -> V {
    // SAFETY: `cross_as_stored!` makes `Tracked` stored as itself.
    unsafe { rt.erase::<Tracked>(drops.payload()) }
}

// -- `Deque<Owned<R>>` ---------------------------------------------------

#[test]
fn a_deque_releases_its_elements_once() {
    let rt = Counted;
    let drops = Drops::default();
    {
        let mut deque = Deque::<Owned<Counted>>::default();
        for _ in 0..3 {
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            deque.push_back(unsafe { Owned::from_value(runtime_holding(), tracked_value(&rt, &drops)) });
        }
        assert_eq!(drops.count(), 0, "the elements are in the deque");
    }
    assert_eq!(drops.count(), 3, "each element was released once");
}

#[test]
fn a_deque_element_popped_is_released_by_its_receiver() {
    let rt = Counted;
    let drops = Drops::default();
    let mut deque = Deque::<Owned<Counted>>::default();
    // SAFETY: the word was made for this holder and moved in; no other holder owns it.
    deque.push_front(unsafe { Owned::from_value(runtime_holding(), tracked_value(&rt, &drops)) });
    let popped = deque
        .pop_front()
        .expect("the element just pushed is the front");
    drop(deque);
    assert_eq!(drops.count(), 0, "the element is out of the deque");
    drop(popped);
    assert_eq!(drops.count(), 1, "its receiver released it once");
}

// -- The owned source ----------------------------------------------------

#[test]
fn an_abandoned_source_releases_the_elements_it_did_not_yield() {
    let rt = Counted;
    let drops = Drops::default();
    let items = (0..3)
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        .map(|_| unsafe { Owned::from_value(runtime_holding(), tracked_value(&rt, &drops)) })
        .collect();
    {
        let _abandoned = Items::<Owned<Counted>, (), Counted>::of(items);
        assert_eq!(drops.count(), 0, "the elements are in the source");
    }
    assert_eq!(drops.count(), 3, "the source released every element once");
}

/// The arguments of a closure call, read back as the run this runtime's own
/// `call_now` reads them from.
fn run_of<Rt, A>(rt: &Rt, args: A) -> Vec<Rt::Value>
where
    Rt: acvus_extern::Runtime,
    A: acvus_extern::IntoRun<Rt>,
{
    let mut run = vec![Rt::Value::default(); A::WIDTH];
    // SAFETY: the test is the runtime, and the arguments cross at the types
    // `A` names.
    args.into_run(unsafe { acvus_extern::Crossing::new(rt) }, &mut run);
    run
}
