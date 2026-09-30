//! The checking reference runtime (runtime contract §1, §3, K15): a runtime
//! that checks every read, whose author writes no `unsafe`. The author
//! implements `CheckedStorage`, which hands out storage as safe Rust objects;
//! `Checked` is the extern side's adapter over it, implements `Runtime` with
//! its own guards and checked reads, and carries the one signature of S1-S3
//! (in the boundary).

use std::any::Any;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::boundary::uniform::{Release, Releasing};
use crate::runtime::{Crosses, ObjectBody, RefTarget, Runtime, TupleBody, VariantBody};

/// One stored value. A Rust value is kept as its canonical form, `'static` by
/// K12, so it is read back by a checked downcast.
pub enum Slot {
    Rust(Box<dyn Any + Send + Sync>),
    Tuple(Box<[u32]>),
    Object(Box<[u32]>),
    Variant { tag: u32, payload: Option<u32> },
    Reference { target: u32, exclusive: bool },
}

/// Storage a checking runtime hands out. Its slots live as long as the
/// storage, at a stable place.
pub trait CheckedStorage: Send + Sync + 'static {
    fn slot(&self, id: u32) -> &RwLock<Slot>;
    fn alloc(&self, slot: Slot) -> u32;
    fn free(&self, id: u32);
    fn interner(&self) -> &Interner;
}

/// The adapter: the contract implemented over a `CheckedStorage`.
pub struct Checked<C: CheckedStorage> {
    storage: &'static C,
}

impl<C: CheckedStorage> Checked<C> {
    pub fn new(storage: &'static C) -> Self {
        Checked { storage }
    }
}

/// A value of the checking runtime: a slot of its storage.
pub struct CheckedValue<C: CheckedStorage> {
    storage: &'static C,
    id: u32,
}

impl<C: CheckedStorage> Clone for CheckedValue<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: CheckedStorage> Copy for CheckedValue<C> {}

impl<C: CheckedStorage> Release for CheckedValue<C> {
    fn release(self, releasing: Releasing) {
        todo!("free the slot, and every slot it owns")
    }
}

pub struct CheckedShared<'a, T> {
    slot: RwLockReadGuard<'a, Slot>,
    _t: PhantomData<fn() -> T>,
}

pub struct CheckedExclusive<'a, T> {
    slot: RwLockWriteGuard<'a, Slot>,
    _t: PhantomData<fn() -> T>,
}

impl<T: Crosses> Deref for CheckedShared<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        todo!("downcast to T::Canon, checked; give it back at T through the layout witness")
    }
}

impl<T: Crosses> Deref for CheckedExclusive<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        todo!("as CheckedShared")
    }
}

impl<T: Crosses> DerefMut for CheckedExclusive<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        todo!("as CheckedShared, exclusively")
    }
}

pub struct CheckedSlice<'a, T> {
    slots: Box<[RwLockReadGuard<'a, Slot>]>,
    _t: PhantomData<fn() -> T>,
}

pub struct CheckedSliceMut<'a, T> {
    slots: Box<[RwLockWriteGuard<'a, Slot>]>,
    _t: PhantomData<fn() -> T>,
}

impl<T: Crosses> Deref for CheckedSlice<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        todo!("a slice view the checking runtime keeps contiguous")
    }
}

impl<T: Crosses> Deref for CheckedSliceMut<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        todo!("as CheckedSlice")
    }
}

impl<T: Crosses> DerefMut for CheckedSliceMut<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        todo!("as CheckedSlice, exclusively")
    }
}

pub struct CheckedStr<'a> {
    slot: RwLockReadGuard<'a, Slot>,
}

impl Deref for CheckedStr<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        todo!("downcast to String, checked")
    }
}

pub struct CheckedParts<C: CheckedStorage> {
    storage: &'static C,
    parts: Vec<Option<u32>>,
}

impl<C: CheckedStorage> TupleBody<CheckedValue<C>> for CheckedParts<C> {
    fn arity(&self) -> u32 {
        todo!()
    }
    fn take_part(&mut self, i: u32) -> CheckedValue<C> {
        todo!("take the part once, checked")
    }
}

impl<C: CheckedStorage> ObjectBody<CheckedValue<C>> for CheckedParts<C> {
    fn len(&self) -> u32 {
        todo!()
    }
    fn take_field(&mut self, i: u32) -> CheckedValue<C> {
        todo!("take the field once, checked")
    }
}

pub struct CheckedVariant<C: CheckedStorage> {
    storage: &'static C,
    tag: u32,
    payload: Option<u32>,
}

impl<C: CheckedStorage> VariantBody<CheckedValue<C>> for CheckedVariant<C> {
    fn tag(&self) -> u32 {
        self.tag
    }
    fn take_payload(&mut self) -> Option<CheckedValue<C>> {
        todo!("take the payload once")
    }
}

impl<C: CheckedStorage> Runtime for Checked<C> {
    type Value = CheckedValue<C>;
    type Tuple = CheckedParts<C>;
    type Object = CheckedParts<C>;
    type Variant = CheckedVariant<C>;
    type Shared<'a, T: Crosses + 'a> = CheckedShared<'a, T>;
    type Exclusive<'a, T: Crosses + 'a> = CheckedExclusive<'a, T>;
    type SharedSlice<'a, T: Crosses + 'a> = CheckedSlice<'a, T>;
    type ExclusiveSlice<'a, T: Crosses + 'a> = CheckedSliceMut<'a, T>;
    type SharedStr<'a> = CheckedStr<'a>;

    fn take<T: Crosses>(&self, v: Self::Value, at: Settled<'_>) -> T {
        todo!("downcast the slot to T::Canon, checked, and take it")
    }
    fn give<T: Crosses>(&self, value: T, at: Settled<'_>) -> Self::Value {
        todo!("store value as T::Canon")
    }
    fn shared<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::Shared<'a, T> {
        todo!("read-lock the slot, checked")
    }
    fn exclusive<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::Exclusive<'a, T> {
        todo!("write-lock the slot, checked")
    }
    fn shared_slice<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::SharedSlice<'a, T> {
        todo!()
    }
    fn exclusive_slice<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::ExclusiveSlice<'a, T> {
        todo!()
    }
    fn shared_str<'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::SharedStr<'a> {
        todo!()
    }
    fn take_tuple(&self, v: Self::Value, at: Settled<'_>) -> Self::Tuple {
        todo!()
    }
    fn take_object(&self, v: Self::Value, at: Settled<'_>) -> Self::Object {
        todo!()
    }
    fn take_variant(&self, v: Self::Value, at: Settled<'_>) -> Self::Variant {
        todo!()
    }
    fn lent_shared<'a, T: Crosses + 'a>(&'a self, guard: Self::Shared<'a, T>) -> Self::Value {
        todo!()
    }
    fn lent_exclusive<'a, T: Crosses + 'a>(&'a self, guard: Self::Exclusive<'a, T>) -> Self::Value {
        todo!()
    }
    fn reference(&self, target: RefTarget<'_, Self::Value>) -> Self::Value {
        todo!()
    }
    fn interner(&self) -> &Interner {
        self.storage.interner()
    }
    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send {
        async move { todo!("a timer") }
    }
}
