//! An unchecked runtime (rfc_extern references §5, second example): a scalar
//! is held in the `Value` itself and every other value behind a pointer to a
//! box that does not move. A shared reference value is the value its place
//! holds, an owning one turned borrowed; an exclusive one is the address of
//! the place. Nothing is checked; a misuse is undefined behaviour, which Miri
//! reports.
//!
//! A `Value` is two words, sixteen bytes. A view of a text or a run is its
//! fat pointer as it is: its start, and its length, whose top bit is clear,
//! since Rust keeps every slice within `isize::MAX` bytes. Every other form
//! sets that bit and states itself in the rest of the second word, which only
//! the operations that take no type read (values §1); every other operation
//! knows its value's kind.
//!
//! How a value is held is the runtime's own (values §1), and two choices are
//! modelled. Where a box is placed: `Heap` gives each box its own
//! allocation, `Frame` places boxes one after another in a block reused
//! frame after frame. Whether a layer is held in its `Value` when it fits:
//! with `Chained`, an option whose payload is a scalar, or such an option, is
//! held in the `Value` as a count of `Some` layers over a scalar or over
//! `None`; its projection lends the option it decodes, and an exclusive one
//! encodes it back.

use std::cell::Cell;
use std::future::Future;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::pin::Pin;
use std::ptr::NonNull;

#[cfg(feature = "count")]
use crate::Counted;
use crate::boundary::{Cross, Generic, TypeVar, Uniform};
use crate::contract::{
    EnumLayout, EnumTable, Enums, Functions, ObjectLayout, ObjectTable, Objects, Opaques, Options, Programs, Results, Runtime, Scalar,
    Scalars, Strings, Structures, Tuples, Views,
};
use crate::laid::{Laid, Table};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Value {
    head: Head,
    /// A view's length; any other form's `HELD`, the form, and, for a scalar
    /// or an option held in its `Value`, its count of `Some` layers.
    meta: usize,
}

/// A value's first word: a scalar's bits, or an address.
#[derive(Clone, Copy)]
#[repr(C)]
union Head {
    bits: u64,
    at: *mut u8,
}

const _: () = assert!(size_of::<usize>() == 8 && size_of::<Value>() == 16, "a value of two words of eight bytes");

/// Set in `meta` by every form but a view's.
const HELD: usize = 1 << (usize::BITS - 1);
/// The count of `Some` layers, in the low half of `meta`.
const SOMES: usize = 0xFFFF_FFFF;
/// A scalar in `bits`, or that many layers of `Some` over it. It owns
/// nothing.
const SCALAR: usize = HELD | 1 << 32;
/// That many layers of `Some` over `None`. It owns nothing.
const OVER_NONE: usize = HELD | 2 << 32;
/// The box at `at`, which this value owns.
const OWNED: usize = HELD | 3 << 32;
/// The box at `at`, borrowed: a shared reference value.
const BORROWED: usize = HELD | 4 << 32;
/// The place at `at`, a `Value`: an exclusive reference value.
const PLACE: usize = HELD | 5 << 32;

/// The form a value's second word states.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    /// A view of a text or a run, `meta` its length: a reference value.
    View,
    Scalar,
    OverNone,
    Owned,
    Borrowed,
    Place,
}

impl Value {
    #[inline]
    fn pointing(at: *mut u8, meta: usize) -> Self {
        Value { head: Head { at }, meta }
    }

    #[inline]
    fn form(self) -> Form {
        match self.meta & !SOMES {
            SCALAR => Form::Scalar,
            OVER_NONE => Form::OverNone,
            OWNED => Form::Owned,
            BORROWED => Form::Borrowed,
            PLACE => Form::Place,
            stated if stated & HELD == 0 => Form::View,
            _ => unreachable!("a value of no form"),
        }
    }

    /// # Safety
    /// The value holds a scalar, or an option over one, in its `Value`.
    #[inline]
    unsafe fn bits(self) -> u64 {
        // SAFETY: the caller's premise.
        unsafe { self.head.bits }
    }

    /// # Safety
    /// The value holds an address: a box, a place, or a view's start.
    #[inline]
    unsafe fn at(self) -> *mut u8 {
        // SAFETY: the caller's premise.
        unsafe { self.head.at }
    }

    /// The count of `Some` layers of an option held in its `Value`.
    #[inline]
    fn somes(self) -> usize {
        self.meta & SOMES
    }
}

/// What every box begins with: how to end the value it holds.
pub struct Header {
    drop: unsafe fn(NonNull<Header>),
}

#[repr(C)]
pub struct Boxed<T> {
    header: Header,
    content: T,
}

/// Where boxes are placed.
pub trait Placement: 'static {
    /// Places `boxed` where it does not move until it is vacated.
    fn place<T>(boxed: Boxed<T>) -> NonNull<Boxed<T>>;

    /// # Safety
    /// `at` was placed by this placement, is live, and is given up here.
    unsafe fn vacate<T>(at: NonNull<Boxed<T>>) -> Boxed<T>;
}

/// Each box its own heap allocation.
pub struct Heap;

impl Placement for Heap {
    fn place<T>(boxed: Boxed<T>) -> NonNull<Boxed<T>> {
        NonNull::from(Box::leak(Box::new(boxed)))
    }

    unsafe fn vacate<T>(at: NonNull<Boxed<T>>) -> Boxed<T> {
        // SAFETY: the caller's premise; `place` leaked this box.
        *unsafe { Box::from_raw(at.as_ptr()) }
    }
}

/// Boxes placed one after another in one block, which `Frame::reset` reuses.
pub struct Frame;

const FRAME_WORDS: usize = 1 << 16;

type Block = [MaybeUninit<u128>; FRAME_WORDS];

/// A thread's frame: one block, freed with the thread.
struct FrameBlock {
    block: NonNull<Block>,
    used: Cell<usize>,
}

impl FrameBlock {
    fn new() -> Self {
        FrameBlock {
            block: NonNull::from(Box::leak(Box::new([const { MaybeUninit::<u128>::uninit() }; FRAME_WORDS]))),
            used: Cell::new(0),
        }
    }

    fn base(&self) -> NonNull<u8> {
        self.block.cast()
    }
}

impl Drop for FrameBlock {
    fn drop(&mut self) {
        // SAFETY: `new` leaked this block, and the thread that placed boxes
        // in it is ending.
        drop(unsafe { Box::from_raw(self.block.as_ptr()) });
    }
}

thread_local! {
    static FRAME: FrameBlock = FrameBlock::new();
}

impl Frame {
    /// Starts the frame over, reusing its block.
    ///
    /// # Safety
    /// No value placed in the frame is live.
    pub unsafe fn reset() {
        FRAME.with(|frame| frame.used.set(0));
    }
}

impl Placement for Frame {
    fn place<T>(boxed: Boxed<T>) -> NonNull<Boxed<T>> {
        const { assert!(align_of::<Boxed<T>>() <= align_of::<u128>(), "a box aligned beyond the frame's block") };
        FRAME.with(|frame| {
            let start = frame.used.get().next_multiple_of(align_of::<Boxed<T>>());
            let end = start + size_of::<Boxed<T>>();
            assert!(end <= FRAME_WORDS * size_of::<u128>(), "the frame is full");
            frame.used.set(end);
            // SAFETY: `start..end` lies inside the block, aligned for `Boxed<T>`.
            let at = unsafe { frame.base().add(start) }.cast::<Boxed<T>>();
            // SAFETY: as above, and nothing else is placed there.
            unsafe { at.write(boxed) };
            at
        })
    }

    unsafe fn vacate<T>(at: NonNull<Boxed<T>>) -> Boxed<T> {
        // SAFETY: the caller's premise; the place is not read again until the
        // frame is reset.
        unsafe { at.read() }
    }
}

/// Whether a layer is held in its `Value` when it fits.
pub trait Layering: 'static {
    const CHAINS: bool;
}

/// An option over a scalar, or over such an option, is held in its `Value`.
pub struct Chained;

impl Layering for Chained {
    const CHAINS: bool = true;
}

/// Every layer that is not a scalar is a box.
pub struct Unchained;

impl Layering for Unchained {
    const CHAINS: bool = false;
}

#[cfg(feature = "count")]
thread_local! {
    static LIVE: Cell<usize> = const { Cell::new(0) };
}

/// Counts a box made, where the model counts them.
fn count_made() {
    #[cfg(feature = "count")]
    LIVE.with(|live| live.set(live.get() + 1));
}

/// Counts a box given up, where the model counts them.
fn count_ended() {
    #[cfg(feature = "count")]
    LIVE.with(|live| live.set(live.get() - 1));
}

fn hold<P, T>(content: T) -> Value
where
    P: Placement,
{
    count_made();
    let at = P::place(Boxed { header: Header { drop: drop_boxed::<P, T> }, content });
    Value::pointing(at.as_ptr().cast(), OWNED)
}

/// # Safety
/// `at` is a box `hold::<P, T>` made, live, and given up by its owner.
unsafe fn drop_boxed<P, T>(at: NonNull<Header>)
where
    P: Placement,
{
    count_ended();
    // SAFETY: the caller's premise.
    drop(unsafe { P::vacate(at.cast::<Boxed<T>>()) });
}

/// # Safety
/// `value` owns a live box made at `T`; the caller owns `value`.
unsafe fn unhold<P, T>(value: Value) -> T
where
    P: Placement,
{
    count_ended();
    // SAFETY: the caller's premise.
    unsafe { P::vacate(NonNull::new_unchecked(value.at()).cast::<Boxed<T>>()) }.content
}

/// # Safety
/// `value` owns or borrows a live box made at `T`, or at a type of its
/// layout (variables §4), not written while the reference lives.
unsafe fn content<'v, T>(value: Value) -> &'v T {
    // SAFETY: the caller's premise.
    unsafe { &(*value.at().cast::<Boxed<T>>()).content }
}

/// # Safety
/// As `content`, the box reached through nothing else while the reference
/// lives.
unsafe fn content_mut<'v, T>(value: Value) -> &'v mut T {
    // SAFETY: the caller's premise.
    unsafe { &mut (*value.at().cast::<Boxed<T>>()).content }
}

/// The value a reference value names: the one in its place, for an
/// exclusive reference; itself, for a shared one.
///
/// # Safety
/// `reference` is a live reference value.
#[inline]
unsafe fn referent(reference: Value) -> Value {
    match reference.meta {
        // SAFETY: the caller's premise: a place holds a live value.
        PLACE => unsafe { *reference.at().cast::<Value>() },
        _ => reference,
    }
}

/// The place an exclusive reference value names.
///
/// # Safety
/// `reference` is a live exclusive reference value, used nowhere else while
/// the place is.
#[inline]
unsafe fn place<'v>(reference: Value) -> &'v mut Value {
    // SAFETY: the caller's premise.
    unsafe { &mut *reference.at().cast::<Value>() }
}

/// The payload of an option held in its `Value`: one `Some` fewer.
#[inline]
fn payload_of(option: Value) -> Value {
    Value { meta: option.meta - 1, ..option }
}

pub struct Boxes<P, L>(PhantomData<(P, L)>);

pub struct BoxScalars<P, L>(PhantomData<(P, L)>);
pub struct BoxStrings<P, L>(PhantomData<(P, L)>);
pub struct BoxTuples<P, L>(PhantomData<(P, L)>);
pub struct BoxOptions<P, L>(PhantomData<(P, L)>);
pub struct BoxResults<P, L>(PhantomData<(P, L)>);
pub struct BoxOpaques<P, L>(PhantomData<(P, L)>);
pub struct BoxObjects<P, L>(PhantomData<(P, L)>);
pub struct BoxEnums<P, L>(PhantomData<(P, L)>);
pub struct BoxFunctions<P, L>(PhantomData<(P, L)>);
pub struct BoxViews<P, L>(PhantomData<(P, L)>);

/// A function value: the program the runtime runs for it.
type FunctionHeld = std::cell::RefCell<Box<dyn FnMut(&[Value]) -> Value>>;

/// A function value called asynchronously: each call gives the future that
/// runs its body.
type AsyncFunctionHeld = std::cell::RefCell<Box<dyn FnMut(&[Value]) -> Pin<Box<dyn Future<Output = Value>>>>>;

/// The runtime's release, as a declared structure's block calls it on each
/// field.
///
/// # Safety
/// As `Runtime::release`.
unsafe fn release_field<P, L>(value: Value)
where
    P: Placement,
    L: Layering,
{
    // SAFETY: the caller's premise.
    unsafe { <Boxes<P, L> as Runtime>::release(value) }
}

/// A part or a payload as a box holds it: a `Value` it owns, at the generic
/// instantiation (variables §3).
type Owned<P, L> = Uniform<Boxes<P, L>, Generic>;

impl<P, L> Runtime for Boxes<P, L>
where
    P: Placement,
    L: Layering,
{
    type Value = Value;

    type Scalars = BoxScalars<P, L>;
    type Strings = BoxStrings<P, L>;
    type Tuples = BoxTuples<P, L>;
    type Options = BoxOptions<P, L>;
    type Results = BoxResults<P, L>;
    type Opaques = BoxOpaques<P, L>;
    type Objects = BoxObjects<P, L>;
    type Enums = BoxEnums<P, L>;
    type Functions = BoxFunctions<P, L>;
    type Views = BoxViews<P, L>;

    unsafe fn release(value: Value) {
        if value.meta == OWNED {
            // SAFETY: the caller owns `value`, which owns a live box.
            let at = unsafe { NonNull::new_unchecked(value.at()) }.cast::<Header>();
            // SAFETY: as above.
            let drop = unsafe { at.as_ref().drop };
            // SAFETY: as above; the box is given up here.
            unsafe { drop(at) }
        }
    }

    // A shared reference to an exclusive one names what that one names, and a
    // place an exclusive reference names never holds another, since one made
    // through another is that one (`value_as_mut`): one step reaches the
    // value. Of the value, only an owner is told apart: a reference to any
    // other form, a view, a borrowed box, or a scalar or an option held in
    // its `Value`, is the value itself, since a reference to a reference is
    // that reference and what is held in a `Value` owns nothing.
    unsafe fn value_as_ref(place: &Value) -> Value {
        // SAFETY: the caller's premise: the place holds a live value or
        // reference value.
        let held = unsafe { referent(*place) };
        match held.meta {
            OWNED => Value { meta: BORROWED, ..held },
            _ => held,
        }
    }

    unsafe fn value_as_mut(place: &mut Value) -> Value {
        match place.form() {
            // A reference to a reference is that reference: an exclusive one,
            // a view, or a shared one, which stays shared.
            Form::Place | Form::View | Form::Borrowed => *place,
            Form::Owned | Form::Scalar | Form::OverNone => Value::pointing(std::ptr::from_mut(place).cast(), PLACE),
        }
    }
}

#[cfg(feature = "count")]
impl<P, L> Counted for Boxes<P, L>
where
    P: Placement,
    L: Layering,
{
    fn live() -> usize {
        LIVE.with(Cell::get)
    }
}

impl<P, L> Scalars<Boxes<P, L>> for BoxScalars<P, L>
where
    P: Placement,
    L: Layering,
{
    fn give<S>(scalar: S) -> Value
    where
        S: Scalar,
    {
        Value { head: Head { bits: scalar.to_bits() }, meta: SCALAR }
    }

    unsafe fn take<S>(value: Value) -> S
    where
        S: Scalar,
    {
        // SAFETY: the caller's premise: made as a scalar.
        S::from_bits(unsafe { value.bits() })
    }

    unsafe fn project<S, O>(reference: Value, lent: impl FnOnce(&S) -> O) -> O
    where
        S: Scalar,
    {
        // SAFETY: the caller's premise.
        lent(&S::from_bits(unsafe { referent(reference).bits() }))
    }

    unsafe fn project_mut<S, O>(reference: Value, lent: impl FnOnce(&mut S) -> O) -> O
    where
        S: Scalar,
    {
        // SAFETY: the caller's premise.
        let place = unsafe { place(reference) };
        // SAFETY: as above, the place holds a scalar.
        let mut scalar = S::from_bits(unsafe { place.bits() });
        let out = lent(&mut scalar);
        place.head = Head { bits: scalar.to_bits() };
        out
    }
}

impl<P, L> Strings<Boxes<P, L>> for BoxStrings<P, L>
where
    P: Placement,
    L: Layering,
{
    fn give(string: String) -> Value {
        hold::<P, _>(string)
    }

    unsafe fn take(value: Value) -> String {
        // SAFETY: the caller's premise.
        unsafe { unhold::<P, _>(value) }
    }

    unsafe fn project<O>(reference: Value, lent: impl FnOnce(&String) -> O) -> O {
        // SAFETY: the caller's premise.
        lent(unsafe { content(referent(reference)) })
    }

    unsafe fn project_mut<O>(reference: Value, lent: impl FnOnce(&mut String) -> O) -> O {
        // SAFETY: the caller's premise.
        lent(unsafe { content_mut(*place(reference)) })
    }
}

impl<P, L> Tuples<Boxes<P, L>> for BoxTuples<P, L>
where
    P: Placement,
    L: Layering,
{
    unsafe fn give<const N: usize>(parts: [Value; N]) -> Value {
        // SAFETY: the caller gives each part up to the tuple, which owns it.
        hold::<P, _>(parts.map(|part| unsafe { Owned::<P, L>::from_value(part) }))
    }

    unsafe fn take<const N: usize>(value: Value) -> [Value; N] {
        // SAFETY: the caller's premise.
        unsafe { unhold::<P, [Owned<P, L>; N]>(value) }.map(Uniform::into_value)
    }

    unsafe fn parts<const N: usize>(reference: Value) -> [Value; N] {
        // SAFETY: the caller's premise; a `Uniform` is `repr(transparent)`
        // over its `Value`.
        let parts = unsafe { &*std::ptr::from_ref(content::<[Owned<P, L>; N]>(referent(reference))).cast::<[Value; N]>() };
        // SAFETY: each part is a live value of the tuple, which the shared
        // reference keeps unchanged.
        parts.each_ref().map(|part| unsafe { <Boxes<P, L> as Runtime>::value_as_ref(part) })
    }

    unsafe fn parts_mut<const N: usize>(reference: Value) -> [Value; N] {
        // SAFETY: the caller's premise.
        let owner = unsafe { *place(reference) };
        // SAFETY: as above; the parts are reached from the box's address, each
        // through its own place.
        let first = unsafe { &raw mut (*owner.at().cast::<Boxed<[Owned<P, L>; N]>>()).content }.cast::<Value>();
        // SAFETY: each place is a distinct live part of the tuple, which the
        // exclusive reference reaches alone.
        std::array::from_fn(|at| unsafe { <Boxes<P, L> as Runtime>::value_as_mut(&mut *first.add(at)) })
    }
}

impl<P, L> Options<Boxes<P, L>> for BoxOptions<P, L>
where
    P: Placement,
    L: Layering,
{
    fn give<Q>(option: Option<Q>) -> Value
    where
        Q: Cross<Boxes<P, L>>,
    {
        match option {
            None if L::CHAINS => Value { head: Head { bits: 0 }, meta: OVER_NONE },
            None => hold::<P, _>(None::<Owned<P, L>>),
            Some(payload) => {
                let payload = payload.give();
                match payload.form() {
                    Form::Scalar | Form::OverNone if L::CHAINS => Value { meta: payload.meta + 1, ..payload },
                    // SAFETY: the payload's value was just made by its own
                    // kind and is the option's.
                    _ => hold::<P, _>(Some(unsafe { Owned::<P, L>::from_value(payload) })),
                }
            }
        }
    }

    unsafe fn take<Q>(value: Value) -> Option<Q>
    where
        Q: Cross<Boxes<P, L>>,
    {
        match value.form() {
            Form::Scalar | Form::OverNone => match value.somes() {
                0 => None,
                // SAFETY: the caller's premise: the payload, given as `Q`, is
                // the caller's along with the option.
                _ => Some(unsafe { Q::take(payload_of(value)) }),
            },
            // SAFETY: as above, the option and its payload held in a box.
            Form::Owned => unsafe { unhold::<P, Option<Owned<P, L>>>(value) }.map(|payload| unsafe { Q::take(payload.into_value()) }),
            Form::Borrowed | Form::Place | Form::View => unreachable!("an option taken by what does not own it"),
        }
    }

    unsafe fn project<Q, O>(reference: Value, lent: impl FnOnce(&Option<Q>) -> O) -> O
    where
        Q: TypeVar<Boxes<P, L>>,
    {
        const { assert!(size_of::<Option<Q>>() == size_of::<Option<Owned<P, L>>>()) };
        // SAFETY: the caller's premise.
        let held = unsafe { referent(reference) };
        match held.form() {
            Form::Scalar | Form::OverNone => match held.somes() {
                0 => lent(&None),
                _ => {
                    // SAFETY: the payload held in the option's `Value`, which
                    // owns nothing, so the projection is never dropped.
                    let projection = ManuallyDrop::new(Some(unsafe { Q::take(payload_of(held)) }));
                    lent(&projection)
                }
            },
            // SAFETY: the caller's premise; `Q` is a `Uniform` (the seal),
            // which differs from `Owned` only in a zero-sized marker
            // (variables §4).
            Form::Owned | Form::Borrowed => lent(unsafe { &*std::ptr::from_ref(content::<Option<Owned<P, L>>>(held)).cast::<Option<Q>>() }),
            Form::Place | Form::View => unreachable!("an option held as no option is"),
        }
    }

    unsafe fn project_mut<Q, O>(reference: Value, lent: impl FnOnce(&mut Option<Q>) -> O) -> O
    where
        Q: TypeVar<Boxes<P, L>>,
    {
        const { assert!(size_of::<Option<Q>>() == size_of::<Option<Owned<P, L>>>()) };
        // SAFETY: the caller's premise.
        let place = unsafe { place(reference) };
        match place.form() {
            Form::Scalar | Form::OverNone => {
                // SAFETY: an option held in its `Value` owns nothing; it is
                // decoded into a projection that owns its payload, and encoded back.
                let mut projection = unsafe { Self::take::<Q>(*place) };
                let out = lent(&mut projection);
                *place = Self::give(projection);
                out
            }
            // SAFETY: as `project`, exclusive.
            Form::Owned => lent(unsafe { &mut *std::ptr::from_mut(content_mut::<Option<Owned<P, L>>>(*place)).cast::<Option<Q>>() }),
            Form::Borrowed | Form::Place | Form::View => unreachable!("an option written in a place that owns none"),
        }
    }
}

impl<P, L> Results<Boxes<P, L>> for BoxResults<P, L>
where
    P: Placement,
    L: Layering,
{
    fn give<T, E>(result: Result<T, E>) -> Value
    where
        T: Cross<Boxes<P, L>>,
        E: Cross<Boxes<P, L>>,
    {
        // SAFETY: as `BoxOptions::give`, for either payload.
        hold::<P, _>(
            result
                .map(|ok| unsafe { Owned::<P, L>::from_value(ok.give()) })
                .map_err(|err| unsafe { Owned::<P, L>::from_value(err.give()) }),
        )
    }

    unsafe fn take<T, E>(value: Value) -> Result<T, E>
    where
        T: Cross<Boxes<P, L>>,
        E: Cross<Boxes<P, L>>,
    {
        // SAFETY: as `BoxOptions::take`, for either payload.
        unsafe { unhold::<P, Result<Owned<P, L>, Owned<P, L>>>(value) }
            .map(|ok| unsafe { T::take(ok.into_value()) })
            .map_err(|err| unsafe { E::take(err.into_value()) })
    }

    unsafe fn project<T, E, O>(reference: Value, lent: impl FnOnce(&Result<T, E>) -> O) -> O
    where
        T: TypeVar<Boxes<P, L>>,
        E: TypeVar<Boxes<P, L>>,
    {
        const { assert!(size_of::<Result<T, E>>() == size_of::<Result<Owned<P, L>, Owned<P, L>>>()) };
        // SAFETY: as `BoxOptions::project`.
        let held = unsafe { content::<Result<Owned<P, L>, Owned<P, L>>>(referent(reference)) };
        // SAFETY: as `BoxOptions::project`.
        lent(unsafe { &*std::ptr::from_ref(held).cast::<Result<T, E>>() })
    }

    unsafe fn project_mut<T, E, O>(reference: Value, lent: impl FnOnce(&mut Result<T, E>) -> O) -> O
    where
        T: TypeVar<Boxes<P, L>>,
        E: TypeVar<Boxes<P, L>>,
    {
        const { assert!(size_of::<Result<T, E>>() == size_of::<Result<Owned<P, L>, Owned<P, L>>>()) };
        // SAFETY: as `BoxOptions::project_mut`.
        let held = unsafe { content_mut::<Result<Owned<P, L>, Owned<P, L>>>(*place(reference)) };
        // SAFETY: as `BoxOptions::project`.
        lent(unsafe { &mut *std::ptr::from_mut(held).cast::<Result<T, E>>() })
    }
}

impl<P, L> Opaques<Boxes<P, L>> for BoxOpaques<P, L>
where
    P: Placement,
    L: Layering,
{
    fn give<X>(value: X) -> Value {
        hold::<P, X>(value)
    }

    unsafe fn take<X>(value: Value) -> X {
        // SAFETY: the caller's premise: made as `X`, or as an instantiation of its type of the
        // same layout (variables §4).
        unsafe { unhold::<P, X>(value) }
    }

    unsafe fn project<X, O>(reference: Value, lent: impl FnOnce(&X) -> O) -> O {
        // SAFETY: as `take`.
        lent(unsafe { content::<X>(referent(reference)) })
    }

    unsafe fn project_mut<X, O>(reference: Value, lent: impl FnOnce(&mut X) -> O) -> O {
        // SAFETY: as `take`, exclusive.
        lent(unsafe { content_mut::<X>(*place(reference)) })
    }
}

impl<P, L> Objects<Boxes<P, L>> for BoxObjects<P, L>
where
    P: Placement,
    L: Layering,
{
    fn give<X>(value: X) -> Value
    where
        X: ObjectLayout,
    {
        // SAFETY: `X::TABLE` states `X`'s layout (`ObjectLayout`'s premise).
        hold::<P, _>(unsafe { Laid::put(value, Table::Object(X::TABLE), release_field::<P, L>) })
    }

    unsafe fn take<X>(value: Value) -> X
    where
        X: ObjectLayout,
    {
        // SAFETY: the caller's premise.
        unsafe { unhold::<P, Laid<Value>>(value).take() }
    }

    unsafe fn project<X, O>(reference: Value, lent: impl FnOnce(&X) -> O) -> O
    where
        X: ObjectLayout,
    {
        // SAFETY: the caller's premise.
        lent(unsafe { content::<Laid<Value>>(referent(reference)).project() })
    }

    unsafe fn project_mut<X, O>(reference: Value, lent: impl FnOnce(&mut X) -> O) -> O
    where
        X: ObjectLayout,
    {
        // SAFETY: the caller's premise.
        lent(unsafe { content_mut::<Laid<Value>>(*place(reference)).project_mut() })
    }
}

impl<P, L> Enums<Boxes<P, L>> for BoxEnums<P, L>
where
    P: Placement,
    L: Layering,
{
    fn give<X>(value: X) -> Value
    where
        X: EnumLayout,
    {
        // SAFETY: `X::TABLE` states `X`'s layout (`EnumLayout`'s premise).
        hold::<P, _>(unsafe { Laid::put(value, Table::Enum(X::TABLE), release_field::<P, L>) })
    }

    unsafe fn take<X>(value: Value) -> X
    where
        X: EnumLayout,
    {
        // SAFETY: the caller's premise.
        unsafe { unhold::<P, Laid<Value>>(value).take() }
    }

    unsafe fn project<X, O>(reference: Value, lent: impl FnOnce(&X) -> O) -> O
    where
        X: EnumLayout,
    {
        // SAFETY: the caller's premise.
        lent(unsafe { content::<Laid<Value>>(referent(reference)).project() })
    }

    unsafe fn project_mut<X, O>(reference: Value, lent: impl FnOnce(&mut X) -> O) -> O
    where
        X: EnumLayout,
    {
        // SAFETY: the caller's premise.
        lent(unsafe { content_mut::<Laid<Value>>(*place(reference)).project_mut() })
    }
}

impl<P, L> Structures for Boxes<P, L>
where
    P: Placement,
    L: Layering,
{
    unsafe fn field(reference: Value, name: &str) -> Value {
        // SAFETY: the caller's premise.
        let field = unsafe { content::<Laid<Value>>(referent(reference)) }.place(name);
        match reference.form() {
            // SAFETY: the field's place, reached through the exclusive
            // reference alone.
            Form::Place => unsafe { Self::value_as_mut(&mut *field.as_ptr()) },
            // SAFETY: the field's place, unchanged while the shared reference
            // lives.
            Form::Borrowed => unsafe { Self::value_as_ref(field.as_ref()) },
            Form::View | Form::Scalar | Form::OverNone | Form::Owned => unreachable!("a field reached through no reference to its structure"),
        }
    }

    unsafe fn variant(reference: Value) -> &'static str {
        // SAFETY: the caller's premise.
        unsafe { content::<Laid<Value>>(referent(reference)) }.variant()
    }

    unsafe fn make_object(table: &'static ObjectTable, fields: &[(&str, Value)]) -> Value {
        // SAFETY: the caller's premise.
        hold::<P, _>(unsafe { Laid::make_object(table, fields, release_field::<P, L>) })
    }

    unsafe fn make_enum(table: &'static EnumTable, variant: &str, fields: &[(&str, Value)]) -> Value {
        // SAFETY: the caller's premise.
        hold::<P, _>(unsafe { Laid::make_enum(table, variant, fields, release_field::<P, L>) })
    }
}

impl<P, L> Functions<Boxes<P, L>> for BoxFunctions<P, L>
where
    P: Placement,
    L: Layering,
{
    unsafe fn call_ref<const N: usize>(function: Value, arguments: [Value; N]) -> Value {
        // SAFETY: the caller's premise.
        let held = unsafe { content::<FunctionHeld>(referent(function)) };
        (held.borrow_mut())(&arguments)
    }

    unsafe fn call_mut<const N: usize>(function: Value, arguments: [Value; N]) -> Value {
        // SAFETY: the caller's premise.
        let held = unsafe { content_mut::<FunctionHeld>(*place(function)) };
        (held.get_mut())(&arguments)
    }

    unsafe fn call_once<const N: usize>(function: Value, arguments: [Value; N]) -> Value {
        // SAFETY: the caller's premise.
        let held = unsafe { unhold::<P, FunctionHeld>(function) };
        (held.into_inner())(&arguments)
    }

    unsafe fn call_async_ref<const N: usize>(function: Value, arguments: [Value; N]) -> impl Future<Output = Value> {
        // SAFETY: the caller's premise.
        let held = unsafe { content::<AsyncFunctionHeld>(referent(function)) };
        (held.borrow_mut())(&arguments)
    }

    unsafe fn call_async_mut<const N: usize>(function: Value, arguments: [Value; N]) -> impl Future<Output = Value> {
        // SAFETY: the caller's premise.
        let held = unsafe { content_mut::<AsyncFunctionHeld>(*place(function)) };
        (held.get_mut())(&arguments)
    }

    unsafe fn call_async_once<const N: usize>(function: Value, arguments: [Value; N]) -> impl Future<Output = Value> {
        // SAFETY: the caller's premise.
        let held = unsafe { unhold::<P, AsyncFunctionHeld>(function) };
        let call = (held.borrow_mut())(&arguments);
        // The call owns its function until it ends: a future dropped between
        // its polls drops the function with it.
        async move {
            let out = call.await;
            drop(held);
            out
        }
    }
}

impl<P, L> Programs for Boxes<P, L>
where
    P: Placement,
    L: Layering,
{
    fn function(program: impl FnMut(&[Value]) -> Value + 'static) -> Value {
        let held: FunctionHeld = std::cell::RefCell::new(Box::new(program));
        hold::<P, _>(held)
    }

    fn async_function(program: impl FnMut(&[Value]) -> Pin<Box<dyn Future<Output = Value>>> + 'static) -> Value {
        let held: AsyncFunctionHeld = std::cell::RefCell::new(Box::new(program));
        hold::<P, _>(held)
    }
}

// A view is its fat pointer as it is: its start, and its length, whose top bit
// is clear since Rust keeps a slice within `isize::MAX` bytes, so it states
// the view's form.
impl<P, L> Views<Boxes<P, L>> for BoxViews<P, L>
where
    P: Placement,
    L: Layering,
{
    fn give_text(text: &str) -> Value {
        Value::pointing(text.as_ptr().cast_mut(), text.len())
    }

    fn give_run<T>(run: &[T]) -> Value
    where
        T: TypeVar<Boxes<P, L>>,
    {
        Value::pointing(run.as_ptr().cast_mut().cast(), run.len())
    }

    fn give_run_mut<T>(run: &mut [T]) -> Value
    where
        T: TypeVar<Boxes<P, L>>,
    {
        Value::pointing(run.as_mut_ptr().cast(), run.len())
    }

    unsafe fn take_text<'a>(view: Value) -> &'a str {
        // SAFETY: the caller's premise: the view's start and length are a
        // live text's.
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(view.at(), view.meta)) }
    }

    unsafe fn take_run<'a, T>(view: Value) -> &'a [T]
    where
        T: TypeVar<Boxes<P, L>>,
    {
        // SAFETY: the caller's premise: the view's start and length are a
        // live run's of `T`'s layout.
        unsafe { std::slice::from_raw_parts(view.at().cast::<T>(), view.meta) }
    }

    unsafe fn take_run_mut<'a, T>(view: Value) -> &'a mut [T]
    where
        T: TypeVar<Boxes<P, L>>,
    {
        // SAFETY: as `take_run`, the run reached through this view alone.
        unsafe { std::slice::from_raw_parts_mut(view.at().cast::<T>(), view.meta) }
    }
}
