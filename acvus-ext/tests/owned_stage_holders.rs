//! RFC-0048 rule 7 at the extension crate's own holders: a `Deque` and an
//! `Iter` stage release what they own, exactly once.
//!
//! The runtime fixture is the one `acvus-extern/tests/owned_holders.rs`
//! uses, repeated here because a test target cannot import another
//! crate's test target.

use std::any::{Any, TypeId, type_name};
use std::future::Ready;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_ext::{Deque, Iter};
use acvus_extern::{
    Astr, CallToken, Fn1, FromValue, Interner, OneValue, Owned, Ref, Release, Runtime,
    cross_as_stored,
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
    type Value = V;
    type Frame = ();
    type CallFuture<'a> = Ready<V>;

    fn frame(&self) {}

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

    fn slice_into_run(&self, _: acvus_extern::Words, _: &mut [Self::Value]) {
        panic!("{NO_SLICES}")
    }

    unsafe fn slice_from_run(&self, _: &[Self::Value]) -> acvus_extern::Words {
        panic!("{NO_SLICES}")
    }

    unsafe fn reference(&self, target: &V) -> V {
        V::Reference(target as *const V)
    }

    fn call_is_sync(&self, _: &V) -> bool {
        true
    }

    fn call_now(&self, f: &V, args: &mut [V], _: &mut (), _: CallToken) -> V {
        let [a] = args else {
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

fn mapping_closure_owning_a_tracked_capture(rt: &Counted, drops: &Drops) -> V {
    let captured = drops.payload();
    let f: UnaryClosure = Box::new(move |rt, argument| {
        argument.release();
        // SAFETY: `usize` is stored as itself.
        unsafe { rt.erase::<usize>(captured.0.load(Ordering::SeqCst)) }
    });
    // SAFETY: a closure is stored as itself.
    unsafe { rt.erase::<UnaryClosure>(f) }
}

fn predicate_closure_owning_a_tracked_capture(rt: &Counted, drops: &Drops) -> V {
    let captured = drops.payload();
    let f: UnaryClosure = Box::new(move |rt, borrowed| {
        assert!(
            matches!(borrowed, V::Reference(_)),
            "a predicate stage lends its element (RFC-0018)"
        );
        // SAFETY: `bool` is stored as itself.
        unsafe { rt.erase::<bool>(captured.0.load(Ordering::SeqCst) == 0) }
    });
    // SAFETY: a closure is stored as itself.
    unsafe { rt.erase::<UnaryClosure>(f) }
}

type Elements = Iter<Owned<Counted>, (), (), Counted>;
type Mapping = Fn1<Owned<Counted>, Owned<Counted>, (), Counted>;
type Predicate = Fn1<Ref<Owned<Counted>, Counted>, bool, (), Counted>;

fn drain(rt: &Counted, mut it: Elements) -> Vec<V> {
    futures::executor::block_on(async {
        let mut out = Vec::new();
        while let Some(value) = it.next_value(rt).await {
            out.push(value);
        }
        out
    })
}

// -- `Deque<Owned<R>>` ---------------------------------------------------

#[test]
fn a_deque_releases_its_elements_once() {
    let rt = Counted;
    let drops = Drops::default();
    {
        let mut deque = Deque::<Owned<Counted>>::default();
        for _ in 0..3 {
            deque.push_back(Owned::from_value(tracked_value(&rt, &drops)));
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
    deque.push_front(Owned::from_value(tracked_value(&rt, &drops)));
    let popped = deque
        .pop_front()
        .expect("the element just pushed is the front");
    drop(deque);
    assert_eq!(drops.count(), 0, "the element is out of the deque");
    drop(popped);
    assert_eq!(drops.count(), 1, "its receiver released it once");
}

// -- The `Iter` stages that own a closure --------------------------------

#[test]
fn a_map_stage_releases_its_closure_when_the_pipeline_ends() {
    let rt = Counted;
    let drops = Drops::default();
    let f = Mapping::new(&rt, mapping_closure_owning_a_tracked_capture(&rt, &drops));
    let yielded = drain(
        &rt,
        Elements::from_items(vec![]).map::<Owned<Counted>>(&rt, f),
    );
    assert!(yielded.is_empty(), "an empty source yields nothing");
    assert_eq!(drops.count(), 1, "the stage released its closure once");
}

#[test]
fn a_filter_stage_releases_its_closure_when_the_pipeline_ends() {
    let rt = Counted;
    let drops = Drops::default();
    let f = Predicate::new(&rt, predicate_closure_owning_a_tracked_capture(&rt, &drops));
    let yielded = drain(&rt, Elements::from_items(vec![]).filter(&rt, f));
    assert!(yielded.is_empty(), "an empty source yields nothing");
    assert_eq!(drops.count(), 1, "the stage released its closure once");
}

#[test]
fn a_take_while_stage_releases_its_closure_when_the_pipeline_ends() {
    let rt = Counted;
    let drops = Drops::default();
    let f = Predicate::new(&rt, predicate_closure_owning_a_tracked_capture(&rt, &drops));
    let yielded = drain(&rt, Elements::from_items(vec![]).take_while(&rt, f));
    assert!(yielded.is_empty(), "an empty source yields nothing");
    assert_eq!(drops.count(), 1, "the stage released its closure once");
}

#[test]
fn a_skip_while_stage_releases_its_closure_when_the_pipeline_ends() {
    let rt = Counted;
    let drops = Drops::default();
    let f = Predicate::new(&rt, predicate_closure_owning_a_tracked_capture(&rt, &drops));
    let yielded = drain(&rt, Elements::from_items(vec![]).skip_while(&rt, f));
    assert!(yielded.is_empty(), "an empty source yields nothing");
    assert_eq!(drops.count(), 1, "the stage released its closure once");
}

// -- The `Iter` source ---------------------------------------------------

#[test]
fn a_drained_pipeline_hands_every_element_to_its_receiver() {
    let rt = Counted;
    let drops = Drops::default();
    let items = (0..3)
        .map(|_| Owned::from_value(tracked_value(&rt, &drops)))
        .collect();
    let yielded = drain(&rt, Elements::from_items(items));
    assert_eq!(yielded.len(), 3, "the source yielded every element");
    assert_eq!(drops.count(), 0, "the elements are out of the pipeline");
    for value in yielded {
        value.release();
    }
    assert_eq!(drops.count(), 3, "each element was released once");
}

#[test]
fn an_undrained_pipeline_releases_the_elements_it_did_not_yield() {
    let rt = Counted;
    let drops = Drops::default();
    let items = (0..3)
        .map(|_| Owned::from_value(tracked_value(&rt, &drops)))
        .collect();
    {
        let _abandoned = Elements::from_items(items).take(1);
    }
    assert_eq!(drops.count(), 3, "the source released every element once");
}
