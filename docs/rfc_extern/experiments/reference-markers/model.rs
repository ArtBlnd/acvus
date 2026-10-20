//! A model of a reference held uniformly, docs/rfc_extern references §4: a
//! sixteen-byte `Value` of two lanes, `Uniform` with a release on drop, `Ref`
//! as a `Copy` type with no drop, and the `Shared` and `Exclusive` markers
//! that let a reference be held uniformly. One runtime, so no `R` parameter.
//!
//! Storage is kept in an arena until `clear_arena`, so a release only counts
//! owners and never frees: a wrong count is observable without undefined
//! behaviour, which is what the negative control needs.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

pub const STORAGE: u64 = 1 << 63;
pub const BORROWED: u64 = (1 << 63) | 1;
pub const PLACE: u64 = (1 << 63) | 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct Value {
    pub p: u64,
    pub m: u64,
}

pub struct Stored {
    pub owners: Cell<isize>,
    pub text: String,
}

thread_local! {
    static ARENA: RefCell<Vec<*mut Stored>> = const { RefCell::new(Vec::new()) };
}

fn hold(text: String) -> Value {
    let stored = Box::into_raw(Box::new(Stored { owners: Cell::new(1), text }));
    let p = stored.expose_provenance() as u64;
    ARENA.with(|arena| arena.borrow_mut().push(stored));
    Value { p, m: STORAGE }
}

/// Frees every stored value. A test calls it once nothing that names storage
/// is left.
pub fn clear_arena() {
    ARENA.with(|arena| {
        for stored in arena.borrow_mut().drain(..) {
            // SAFETY: each pointer came from `Box::into_raw` and is freed once.
            drop(unsafe { Box::from_raw(stored) });
        }
    });
}

/// # Safety
///
/// `value` names storage made by `hold` on this thread, not yet cleared.
unsafe fn stored<'s>(value: &Value) -> &'s Stored {
    // SAFETY: the arena keeps every `Stored` until `clear_arena`.
    unsafe { &*std::ptr::with_exposed_provenance::<Stored>(value.p as usize) }
}

/// The owner count of the storage `value` names.
pub fn owners(value: Value) -> isize {
    // SAFETY: only called on values made by `hold`.
    unsafe { stored(&value) }.owners.get()
}

/// The runtime's untyped release: it reads the `Value`, and ends an owner only
/// where the lanes say the value owns storage.
pub fn release(value: Value) {
    if value.m == STORAGE {
        // SAFETY: a storage marker is written by `hold` alone.
        let stored = unsafe { stored(&value) };
        stored.owners.set(stored.owners.get() - 1);
    }
}

/// The runtime's untyped `as_ref`: an owning `Value` turned borrowed, any
/// other left as it is.
fn borrow(value: &Value) -> Value {
    match value.m {
        STORAGE => Value { p: value.p, m: BORROWED },
        _ => *value,
    }
}

pub trait Crossing: Sized {
    fn give(self) -> Value;
}

impl Crossing for i64 {
    fn give(self) -> Value {
        Value { p: 0, m: self as u64 }
    }
}

impl Crossing for String {
    fn give(self) -> Value {
        hold(self)
    }
}

/// A value type's projection, read through the runtime.
pub trait Lend: Crossing {
    fn project<O>(value: &Value, lent: impl FnOnce(&Self) -> O) -> O;
}

impl Lend for i64 {
    fn project<O>(value: &Value, lent: impl FnOnce(&i64) -> O) -> O {
        lent(&(value.m as i64))
    }
}

impl Lend for String {
    fn project<O>(value: &Value, lent: impl FnOnce(&String) -> O) -> O {
        // SAFETY: a `Value` lent as a `String` names storage `hold` made.
        lent(&unsafe { stored(value) }.text)
    }
}

pub enum Never {}

/// A reference to `X` held uniformly, its lifetime in the type.
pub struct Shared<'a, X>(PhantomData<(fn(X) -> X, fn() -> &'a ())>);

/// An exclusive reference to `X` held uniformly.
pub struct Exclusive<'a, X>(PhantomData<(fn(X) -> X, fn() -> &'a ())>);

#[repr(transparent)]
pub struct Uniform<X> {
    value: Value,
    marker: PhantomData<fn(X) -> X>,
}

impl<X> Drop for Uniform<X> {
    fn drop(&mut self) {
        release(self.value);
    }
}

impl<X> Uniform<X> {
    pub fn bits(&self) -> Value {
        self.value
    }
}

impl<X> Uniform<X>
where
    X: Lend,
{
    pub fn new(value: X) -> Self
    where
        X: 'static,
    {
        Uniform { value: value.give(), marker: PhantomData }
    }

    pub fn as_ref(&self) -> Ref<'_, X> {
        Ref { value: borrow(&self.value), marker: PhantomData }
    }

    pub fn as_mut(&mut self) -> Mut<'_, X> {
        let p = std::ptr::from_mut(&mut self.value).expose_provenance() as u64;
        Mut { value: Value { p, m: PLACE }, marker: PhantomData }
    }

    pub fn with<O>(&self, lent: impl FnOnce(&X) -> O) -> O {
        X::project(&self.value, lent)
    }

    /// The negative control: a `Ref` that copies the owning `Value` bit for
    /// bit, as the note had it before the borrowed marker.
    pub fn as_ref_verbatim(&self) -> Ref<'_, X> {
        Ref { value: self.value, marker: PhantomData }
    }
}

impl<'a, X> Uniform<Shared<'a, X>> {
    pub fn from_ref(reference: Ref<'a, X>) -> Self {
        Uniform { value: reference.value, marker: PhantomData }
    }

    pub fn get(&self) -> Ref<'a, X> {
        Ref { value: self.value, marker: PhantomData }
    }
}

impl<'a, X> Uniform<Exclusive<'a, X>> {
    pub fn from_mut(reference: Mut<'a, X>) -> Self {
        Uniform { value: reference.value, marker: PhantomData }
    }

    pub fn as_mut(&mut self) -> Mut<'_, X> {
        Mut { value: self.value, marker: PhantomData }
    }
}

/// The one bound of a type variable. It holds for every `Uniform<X>`.
pub trait TypeVar {
    fn as_ref_var(&self) -> Ref<'_, Self>
    where
        Self: Sized;
}

impl<X> TypeVar for Uniform<X> {
    fn as_ref_var(&self) -> Ref<'_, Self> {
        Ref { value: borrow(&self.value), marker: PhantomData }
    }
}

/// The one bound of an effect variable.
pub trait EffectVar {}

/// The effect marker an entry instantiates an effect variable at.
pub enum Effect0 {}

impl EffectVar for Effect0 {}

#[repr(transparent)]
pub struct Ref<'a, X> {
    value: Value,
    marker: PhantomData<(fn(X) -> X, fn() -> &'a ())>,
}

impl<X> Clone for Ref<'_, X> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<X> Copy for Ref<'_, X> {}

impl<'a, X> Ref<'a, X> {
    pub fn bits(self) -> Value {
        self.value
    }
}

impl<'a, X> Ref<'a, X>
where
    X: Lend,
{
    pub fn with<O>(self, lent: impl FnOnce(&X) -> O) -> O {
        X::project(&self.value, lent)
    }
}

#[repr(transparent)]
pub struct Mut<'a, X> {
    value: Value,
    marker: PhantomData<(fn(X) -> X, fn() -> &'a ())>,
}

impl Mut<'_, i64> {
    pub fn set(&mut self, scalar: i64) {
        let place = std::ptr::with_exposed_provenance_mut::<Value>(self.value.p as usize);
        // SAFETY: a `Mut` names a live place, held exclusively for `'_`.
        unsafe { *place = scalar.give() };
    }
}

/// Reads a `Vec` at one element instantiation as another, as one value is
/// read at several instantiations. Sound by `Vec::from_raw_parts` where the element types have
/// one size and alignment, which `repr(transparent)` over `Value` gives every
/// `Uniform<X>`; it does not touch the open question of `Vec<A>` against
/// `Vec<B>` as types, which layout.rs probes.
pub fn read_as<A, B>(items: Vec<A>) -> Vec<B> {
    const { assert!(size_of::<A>() == size_of::<B>() && align_of::<A>() == align_of::<B>()) };
    let mut items = std::mem::ManuallyDrop::new(items);
    let (ptr, len, cap) = (items.as_mut_ptr(), items.len(), items.capacity());
    // SAFETY: same size and alignment, and the buffer is not used again as `A`.
    unsafe { Vec::from_raw_parts(ptr.cast::<B>(), len, cap) }
}
