//! The one module that writes `unsafe`: reading untyped storage back as a
//! Rust type. Every read here rests on the compiler's premise that a value is
//! read at the type it was made at, or at one of the same layout (the extern
//! note: Type variables). Nothing here checks that premise; a safe runtime
//! checks what it can around these reads, in its own storage.

#![allow(unsafe_code, clippy::disallowed_methods)]

use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering, fence};

use crate::kind::{OpaqueKind, RefKind};
use crate::runtime::{Runtime, Stores};
use crate::uniform::Uniform;

/// Reads `from` as `To`, a reference cast between two types that differ only
/// in `Value`s seen as `Uniform`s: `Uniform<R, X>` is `R::Value` behind
/// `repr(transparent)`, and the layouts' sizes and alignments are checked
/// where each cast is instantiated. That the two are laid out alike beyond
/// that is the note's premise for `X<T>` and `X<U>`, which Miri checks.
fn reread<From, To>(from: &From) -> &To {
    const { assert!(size_of::<From>() == size_of::<To>() && align_of::<From>() == align_of::<To>(), "a uniform view keeps its layout") };
    // SAFETY: the premise above; the borrow is carried over unchanged.
    unsafe { &*std::ptr::from_ref(from).cast::<To>() }
}

fn reread_mut<From, To>(from: &mut From) -> &mut To {
    const { assert!(size_of::<From>() == size_of::<To>() && align_of::<From>() == align_of::<To>(), "a uniform view keeps its layout") };
    // SAFETY: as `reread`; the exclusive borrow is carried over unchanged.
    unsafe { &mut *std::ptr::from_mut(from).cast::<To>() }
}

pub(crate) fn uniform_option<R, Y>(option: &Option<R::Value>) -> &Option<Uniform<R, Y>>
where
    R: Runtime,
{
    reread(option)
}

pub(crate) fn uniform_option_mut<R, Y>(option: &mut Option<R::Value>) -> &mut Option<Uniform<R, Y>>
where
    R: Runtime,
{
    reread_mut(option)
}

pub(crate) fn uniform_result<R, Y, E>(result: &Result<R::Value, R::Value>) -> &Result<Uniform<R, Y>, Uniform<R, E>>
where
    R: Runtime,
{
    reread(result)
}

pub(crate) fn uniform_result_mut<R, Y, E>(
    result: &mut Result<R::Value, R::Value>,
) -> &mut Result<Uniform<R, Y>, Uniform<R, E>>
where
    R: Runtime,
{
    reread_mut(result)
}

#[repr(C)]
struct Header {
    count: AtomicUsize,
    /// Drops the value and frees the block this header heads.
    end: unsafe fn(NonNull<Header>),
}

#[repr(C)]
struct Block<X> {
    header: Header,
    value: X,
}

/// One value of any type, its type erased, behind one pointer to a header
/// and the value. The header counts the value's owners and knows how to drop
/// it, so the type needs no `'static`: a value that holds a loan is kept here
/// as it is.
pub struct Boxed {
    header: NonNull<Header>,
}

// SAFETY: `Boxed::new` takes only a value that is `Send + Sync`, and the
// count is atomic, so an owner may move to another thread and owners on two
// threads may share the value.
unsafe impl Send for Boxed {}
unsafe impl Sync for Boxed {}

/// # Safety
///
/// `header` heads a `Block<X>` that `Boxed::new::<X>` allocated, and its
/// count has reached zero.
unsafe fn end<X>(header: NonNull<Header>) {
    // SAFETY: the block was allocated as a `Box<Block<X>>`, and no owner is
    // left to read it.
    drop(unsafe { Box::from_raw(header.cast::<Block<X>>().as_ptr()) });
}

impl Boxed {
    pub fn new<X>(value: X) -> Self
    where
        X: Send + Sync,
    {
        let block = Box::new(Block { header: Header { count: AtomicUsize::new(1), end: end::<X> }, value });
        Boxed { header: NonNull::from(Box::leak(block)).cast() }
    }

    fn header(&self) -> &Header {
        // SAFETY: the block lives while any owner does, and this is one.
        unsafe { self.header.as_ref() }
    }

    /// Whether this is the value's only owner, so that a runtime can check
    /// before it gives the value up to be taken.
    pub fn is_unique(&self) -> bool {
        self.header().count.load(Ordering::Acquire) == 1
    }

    /// A second owner of the same value.
    pub fn share(&self) -> Self {
        // As `Arc::clone`: a new owner is made only from an existing one, so
        // no ordering is needed to keep the block alive.
        self.header().count.fetch_add(1, Ordering::Relaxed);
        Boxed { header: self.header }
    }

    /// # Safety
    ///
    /// The value was made as `X`, or as a type of `X`'s layout.
    pub unsafe fn as_ref<X>(&self) -> &X {
        // SAFETY: the caller's premise; the block is alive while `self` is.
        unsafe { &self.header.cast::<Block<X>>().as_ref().value }
    }

    /// # Safety
    ///
    /// As `as_ref`, and no other owner reads the value while this lives.
    pub unsafe fn as_mut<X>(&mut self) -> &mut X {
        // SAFETY: the caller's premise.
        unsafe { &mut self.header.cast::<Block<X>>().as_mut().value }
    }

    /// # Safety
    ///
    /// As `as_ref`, and this is the value's only owner.
    pub unsafe fn into_inner<X>(self) -> X {
        let header = self.header;
        std::mem::forget(self);
        // SAFETY: the caller's premise: the block is a `Block<X>` and no
        // other owner will drop it.
        let block = unsafe { Box::from_raw(header.cast::<Block<X>>().as_ptr()) };
        let Block { value, .. } = *block;
        value
    }
}

impl Drop for Boxed {
    fn drop(&mut self) {
        let count = &self.header().count;
        // A value never shared is ended after one load: an owner that sees a
        // count of one is the only one, since only an owner makes another.
        let last = count.load(Ordering::Acquire) == 1 || {
            let last = count.fetch_sub(1, Ordering::Release) == 1;
            if last {
                fence(Ordering::Acquire);
            }
            last
        };
        if last {
            let end = self.header().end;
            // SAFETY: the header was made with the `end` of its own block's
            // type, and this was its last owner.
            unsafe { end(self.header) }
        }
    }
}

/// A reference's two words. Making them reads nothing; reading them back as
/// a reference is `unsafe`. The words hold the pointer's own bytes in two
/// pointer-sized slots, so the pointer's provenance travels with them, and
/// nothing assumes which slot of a fat pointer holds its address and which
/// its metadata.
#[derive(Clone, Copy)]
pub struct Words([MaybeUninit<*const ()>; 2]);

// SAFETY: `Words` is data; reading it back as a reference is `unsafe`, and
// that read is where any thread's claim on the place is answered.
unsafe impl Send for Words {}
unsafe impl Sync for Words {}

impl Words {
    pub fn of<X>(pointer: *const X) -> Self
    where
        X: ?Sized,
    {
        const { assert!(size_of::<*const X>() <= size_of::<[MaybeUninit<*const ()>; 2]>(), "a reference is at most two words") };
        let mut words = [MaybeUninit::uninit(); 2];
        // SAFETY: both are valid for the pointer's size, which fits the
        // words by the assertion above, and they do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping((&raw const pointer).cast::<u8>(), words.as_mut_ptr().cast::<u8>(), size_of::<*const X>());
        }
        Words(words)
    }

    /// # Safety
    ///
    /// The words were made by `of::<X>`, or for a type of `X`'s pointer
    /// layout.
    pub unsafe fn pointer<X>(self) -> *const X
    where
        X: ?Sized,
    {
        let mut pointer = MaybeUninit::<*const X>::uninit();
        // SAFETY: the words hold the bytes `of` copied from a `*const X`.
        unsafe {
            std::ptr::copy_nonoverlapping(self.0.as_ptr().cast::<u8>(), pointer.as_mut_ptr().cast::<u8>(), size_of::<*const X>());
            pointer.assume_init()
        }
    }
}

/// The Opaque kind for any runtime that `Stores`: a value lives in a
/// `Boxed`, and is read back at the type the compiler settled.
pub struct StoredOpaque;

impl<R> OpaqueKind<R> for StoredOpaque
where
    R: Stores,
{
    fn make<X>(rt: &R, value: X) -> R::Value
    where
        X: Send + Sync,
    {
        rt.hold(Boxed::new(value))
    }

    fn take<X>(rt: &R, value: R::Value) -> X {
        // SAFETY: the compiler's premise: the value was made as `X`, and a
        // value taken by value has no other owner.
        unsafe { rt.unhold(value).into_inner() }
    }

    fn view<X, O>(rt: &R, value: &R::Value, lent: impl FnOnce(&X) -> O) -> O {
        // SAFETY: the compiler's premise: the value was made as `X`.
        rt.lend(value, |boxed| lent(unsafe { boxed.as_ref() }))
    }

    fn view_mut<X, O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut X) -> O) -> O {
        // SAFETY: the compiler's premise: the value was made as `X`, and the
        // runtime lends it exclusively.
        rt.lend_mut(value, |boxed| lent(unsafe { boxed.as_mut() }))
    }
}

/// The Ref kind for any runtime that `Stores`: a reference value holds the
/// reference's words, read back as a reference only within a view.
pub struct StoredRef;

impl<R> RefKind<R> for StoredRef
where
    R: Stores,
{
    fn make<X>(rt: &R, reference: &X) -> R::Value
    where
        X: ?Sized,
    {
        rt.hold_words(Words::of(std::ptr::from_ref(reference)))
    }

    fn make_mut<X>(rt: &R, reference: &mut X) -> R::Value
    where
        X: ?Sized,
    {
        rt.hold_words(Words::of(std::ptr::from_mut(reference).cast_const()))
    }

    fn view<X, O>(rt: &R, value: &R::Value, lent: impl FnOnce(&X) -> O) -> O
    where
        X: ?Sized,
    {
        let pointer = rt.words(value);
        // SAFETY: the compiler's premise: the words name a live `X` that
        // nothing writes while this view lives (rfc_compiler ownership §5).
        lent(unsafe { &*pointer.pointer::<X>() })
    }

    fn view_mut<X, O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut X) -> O) -> O
    where
        X: ?Sized,
    {
        let pointer = rt.words(value);
        // SAFETY: the compiler's premise: the words were made from an
        // exclusive reference to a live `X`, reached only through this value
        // while the view lives (rfc_compiler ownership §5).
        lent(unsafe { &mut *pointer.pointer::<X>().cast_mut() })
    }
}
