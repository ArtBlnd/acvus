//! The interpreter's own casts (RFC-0080 rule 5): where a register lies in
//! a frame and a frame's cells as its registers, a `Large`'s allocation and
//! the vtable that frees it, and a closure record's trailing captures. These
//! are facts of this runtime's storage, which `acvus_extern::repr` does not
//! hold; that module keeps the casts the extern contract makes. Outside the
//! two, `repr_boundary` refuses a cast.

#![allow(
    clippy::disallowed_methods,
    reason = "the boundary module: a cast the workspace's lints refuse elsewhere is written here"
)]

use std::alloc::{self, Layout};
use std::any::TypeId;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr::{self, NonNull};
use std::slice;

use acvus_extern::repr::{self as word, PtrWord};
use acvus_extern::{Holding, Owned};

use crate::regs::{CELL_SLOTS, Cell};
use crate::runtime::AcvusRuntime;
use crate::value::{Kind, Value};
use crate::vtable::{Composite, NameFn};

// -- A slot's place in a run ------------------------------------------------

/// An index below `N`.
///
/// Its field is private, so every `Below<N>` was made here, by a constructor
/// that holds it below `N`: `new` and `of` check the index, and `first`
/// checks a run's length once; `masked`, `compose` and `step_to` take the
/// bound from their arguments' types and a constant assertion, and check
/// nothing at run time. `Disp::of_below` and `Disp::after` read the bound
/// from the type, so they multiply without a check.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Below<const N: u16>(u16);

impl<const N: u16> Below<N> {
    #[inline]
    pub const fn new(index: u16) -> Option<Below<N>> {
        match index < N {
            true => Some(Below(index)),
            false => None,
        }
    }

    /// # Panics
    /// `index` is not below `N`. In a constant this is a compile error.
    #[inline]
    pub const fn of(index: u16) -> Below<N> {
        let Some(below) = Below::new(index) else {
            panic!("Below::of: the index is not below its bound")
        };
        below
    }

    /// The first `len` indices, in order.
    ///
    /// # Panics
    /// `len` is above `N`.
    pub fn first(len: usize) -> impl Iterator<Item = Below<N>> {
        let len = u16::try_from(len)
            .ok()
            .filter(|len| *len <= N)
            .unwrap_or_else(|| {
                panic!("Below::first: a run of {len} indices is past the bound {N}")
            });
        (0..len).map(Below)
    }

    /// The low bits of `bits` that name an index below `N`, a power of two:
    /// `bits & (N - 1)`. A value already below `N`, such as a nonzero `u64`'s
    /// `trailing_zeros` for `N = 64`, is itself.
    #[inline(always)]
    pub const fn masked(bits: u32) -> Below<N> {
        const {
            assert!(
                N.is_power_of_two(),
                "Below::masked: a mask names every index below a power of two alone"
            )
        };
        Below((bits & (N as u32 - 1)) as u16)
    }

    /// `hi * B + lo`, which is at most `(W - 1) * B + B - 1 = W * B - 1`.
    #[inline(always)]
    pub const fn compose<const W: u16, const B: u16>(hi: Below<W>, lo: Below<B>) -> Below<N> {
        const {
            assert!(
                W as u32 * B as u32 <= N as u32,
                "Below::compose: W * B indices do not all lie below N"
            )
        };
        Below(hi.0 * B + lo.0)
    }

    /// The index after this one, when this one is before `last`: then it is
    /// at most `last`, which is below `N`.
    #[inline(always)]
    pub const fn step_to(self, last: Below<N>) -> Option<Below<N>> {
        match self.0 < last.0 {
            true => Some(Below(self.0 + 1)),
            false => None,
        }
    }

    #[inline(always)]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// The place of one slot of a run of `S`s: its byte displacement from the
/// run's first byte, `i * size_of::<S>()` for slot `i` (RFC-0080 rule 5).
///
/// It is multiplied once, where it is made, so a read at it adds and scales
/// nothing (RFC-0052 rule 5). It is a `u16`, and every `Disp<S>` is a whole
/// number of `S`s that fits one, so no reader checks it again.
#[repr(transparent)]
pub struct Disp<S>(u16, PhantomData<fn() -> S>);

impl<S> Clone for Disp<S> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Copy for Disp<S> {}

impl<S> PartialEq for Disp<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<S> Eq for Disp<S> {}

impl<S> PartialOrd for Disp<S> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<S> Ord for Disp<S> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<S> std::hash::Hash for Disp<S> {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.0.hash(state);
    }
}

impl<S> fmt::Debug for Disp<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Disp").field(&self.0).finish()
    }
}

impl<S> Disp<S> {
    /// Slot `index` of a run of `S`s.
    ///
    /// # Panics
    /// `index * size_of::<S>()` is above `u16::MAX`. In a constant this is a
    /// compile error.
    #[inline]
    pub const fn of(index: u16) -> Disp<S> {
        let Some(byte) = index.checked_mul(Self::SLOT) else {
            panic!("Disp::of: the slot's displacement does not fit the u16 a Disp holds")
        };
        Disp(byte, PhantomData)
    }

    /// Slot `index` of a run of `S`s, below `N`.
    #[inline(always)]
    pub const fn of_below<const N: u16>(index: Below<N>) -> Disp<S> {
        const { Self::fits::<N>() };
        Disp(index.0 * Self::SLOT, PhantomData)
    }

    /// The slot after `index`: at most slot `N`, whose displacement `fits`
    /// holds in the `u16`.
    #[inline(always)]
    pub const fn after<const N: u16>(index: Below<N>) -> Disp<S> {
        const { Self::fits::<N>() };
        Disp((index.0 + 1) * Self::SLOT, PhantomData)
    }

    const fn fits<const N: u16>() {
        assert!(
            N as usize * mem::size_of::<S>() <= u16::MAX as usize,
            "Disp: the displacement of a Below's bound does not fit the u16 a Disp holds"
        );
    }

    const SLOT: u16 = {
        assert!(
            mem::size_of::<S>() != 0 && mem::size_of::<S>() <= u16::MAX as usize,
            "Disp: a slot is at least one byte and at most u16::MAX bytes wide"
        );
        mem::size_of::<S>() as u16
    };

    #[inline(always)]
    pub const fn byte(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        (self.0 / Self::SLOT) as usize
    }
}

/// The displacement of a slot below `N` of a run of `S`s. It is the `Disp` a
/// holder reads, and it keeps the bound it was made under, so the slot after
/// it is a displacement with no check: `after` divides its own displacement
/// back into the index, which `of` made from a `Below<N>`.
#[repr(transparent)]
pub struct DispBelow<S, const N: u16>(Disp<S>);

impl<S, const N: u16> Clone for DispBelow<S, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S, const N: u16> Copy for DispBelow<S, N> {}

impl<S, const N: u16> PartialEq for DispBelow<S, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<S, const N: u16> Eq for DispBelow<S, N> {}

impl<S, const N: u16> fmt::Debug for DispBelow<S, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DispBelow").field(&self.0.0).finish()
    }
}

impl<S, const N: u16> DispBelow<S, N> {
    #[inline(always)]
    pub const fn of(index: Below<N>) -> DispBelow<S, N> {
        DispBelow(Disp::of_below(index))
    }

    #[inline(always)]
    pub const fn disp(self) -> Disp<S> {
        self.0
    }

    #[inline(always)]
    pub const fn index(self) -> Below<N> {
        Below(self.0.0 / Disp::<S>::SLOT)
    }

    /// The slot after this one.
    #[inline(always)]
    pub const fn after(self) -> Disp<S> {
        Disp::after(self.index())
    }
}

/// The slot `d` names in the run of `S`s that begins at `base`.
///
/// # Safety
/// `base` is the first slot of a run of `S`s that lies in one allocation and
/// has the slot `d` names.
#[inline(always)]
pub unsafe fn at<S>(base: NonNull<S>, d: Disp<S>) -> NonNull<S> {
    // SAFETY: the caller's contract puts the slot inside `base`'s allocation,
    // and `d` is a whole number of `S`s (`Disp::of`), so the result is aligned
    // as `base` is.
    unsafe { base.byte_add(d.byte()) }
}

/// The `u64` at byte `byte` of the allocation `base` points into: a mark word
/// laid past a frame's registers, or the word of a register a chain leaf
/// names.
///
/// # Safety
/// The eight bytes at `byte` past `base` lie inside the allocation `base`
/// points into, and their first is aligned to a `u64`.
#[inline(always)]
pub unsafe fn word_at<S>(base: NonNull<S>, byte: usize) -> NonNull<u64> {
    // SAFETY: the caller's contract: the word lies inside `base`'s allocation
    // and is aligned to a `u64`.
    unsafe { base.byte_add(byte) }.cast::<u64>()
}

// -- A head with a trailing array -------------------------------------------

/// The layout of one allocation that holds an `H` and then a run of `E`s:
/// `Layout::extend`'s, so the run starts at `E`'s own alignment whatever
/// `H`'s size and alignment are on the target. The run's length is a `u16`,
/// and the widest such record is a valid layout (the assertion in `TAIL`).
pub struct HeadAndTail<H, E>(PhantomData<fn() -> (H, E)>);

impl<H, E> HeadAndTail<H, E> {
    /// The byte offset of the first `E`: `H`'s size rounded up to `E`'s
    /// alignment, which is what `Layout::extend` gives.
    pub const TAIL: usize = {
        let Ok((_, tail)) = Layout::new::<H>().extend(Layout::new::<E>()) else {
            panic!("a head and one element of its tail are a valid layout")
        };
        assert!(
            tail + (u16::MAX as usize) * mem::size_of::<E>() + mem::align_of::<H>() + mem::align_of::<E>()
                < isize::MAX as usize,
            "a record of the widest tail a u16 can count is a valid layout"
        );
        tail
    };

    /// The layout of a record whose tail is `len` elements, padded to its
    /// alignment.
    #[inline]
    pub fn layout(len: u16) -> Layout {
        let head = Layout::new::<H>();
        let tail = Layout::new::<E>();
        let align = if head.align() > tail.align() { head.align() } else { tail.align() };
        let size = Self::TAIL + usize::from(len) * tail.size();
        // SAFETY: `align` is one of two layouts' alignments, a power of two,
        // and `TAIL`'s assertion bounds `size` rounded up to it below
        // `isize::MAX`.
        unsafe { Layout::from_size_align_unchecked(size, align) }.pad_to_align()
    }

    /// The first element of the tail of the record `head` begins.
    ///
    /// # Safety
    /// `head` is the start of an allocation of `layout(len)` for some `len`.
    #[inline(always)]
    pub unsafe fn tail(head: NonNull<H>) -> NonNull<E> {
        // SAFETY: the caller's contract: the allocation reaches `TAIL`.
        unsafe { head.byte_add(Self::TAIL) }.cast::<E>()
    }

    /// A record of a tail of `len` elements, its head and its tail
    /// unwritten.
    pub fn alloc(len: u16) -> NonNull<H> {
        const {
            assert!(
                mem::size_of::<H>() != 0,
                "a record's head has a size, so the record's allocation has one"
            )
        };
        let layout = Self::layout(len);
        // SAFETY: the layout is at least the head's size, which is not zero.
        let block = unsafe { alloc::alloc(layout) }.cast::<H>();
        match NonNull::new(block) {
            Some(head) => head,
            None => alloc::handle_alloc_error(layout),
        }
    }

    /// The record's allocation, given back.
    ///
    /// # Safety
    /// `head` is `alloc(len)`'s, what its head and tail hold is dropped or
    /// moved out, and nothing names the record after this.
    pub unsafe fn dealloc(head: NonNull<H>, len: u16) {
        // SAFETY: the caller's contract: the allocation is `layout(len)`'s.
        unsafe { alloc::dealloc(head.as_ptr().cast::<u8>(), Self::layout(len)) }
    }

    /// The tail of `len` elements the record `head` begins, read for `'a`.
    ///
    /// # Safety
    /// `head` is `alloc(len)`'s, each of its tail's `len` elements is
    /// written, and the record is live and written through no other name for
    /// `'a`.
    #[inline(always)]
    pub unsafe fn tail_run<'a>(head: NonNull<H>, len: u16) -> &'a [E] {
        // SAFETY: the caller's contract, and `layout(len)` holds `len`
        // elements from `TAIL`.
        unsafe { slice::from_raw_parts(Self::tail(head).as_ptr(), usize::from(len)) }
    }

    /// As `tail_run`, exclusively.
    ///
    /// # Safety
    /// As `tail_run`'s, and the result is the record's only name used for
    /// `'a`.
    #[inline(always)]
    pub unsafe fn tail_run_mut<'a>(head: NonNull<H>, len: u16) -> &'a mut [E] {
        // SAFETY: as `tail_run`'s, exclusively.
        unsafe { slice::from_raw_parts_mut(Self::tail(head).as_ptr(), usize::from(len)) }
    }
}

// -- A head at byte 0 ---------------------------------------------------------

/// The witness that an `H` lies at byte 0 of every `P`, so a pointer to a
/// `P` is a pointer to its `H`, and a pointer to an `H` that is a `P`'s head
/// is a pointer to that `P`.
///
/// Only `head_at_zero!` makes one: it asserts, at each instantiation, that
/// the named field is at offset 0 and is of type `H`.
pub struct Prefix<P, H>(PhantomData<(fn(P) -> P, fn(H) -> H)>);

impl<P, H> Clone for Prefix<P, H> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, H> Copy for Prefix<P, H> {}

macro_rules! head_at_zero {
    ($p:ty, $field:ident: $h:ty) => {{
        const {
            assert!(
                mem::offset_of!($p, $field) == 0,
                "a prefix's head lies at byte 0 of the whole"
            )
        };
        let _field_is_the_head: fn(&$p) -> &$h = |whole| &whole.$field;
        Prefix::<$p, $h>(PhantomData)
    }};
}

impl<P, H> Prefix<P, H> {
    /// The head of the `P` at `whole`.
    #[inline(always)]
    pub fn head(self, whole: NonNull<P>) -> NonNull<H> {
        whole.cast::<H>()
    }

    /// The `P` whose head is at `head`.
    ///
    /// # Safety
    /// The `H` at `head` is the head of a `P`, and `head` was made from a
    /// pointer to that whole `P` (`Prefix::head`), so it reaches all of it.
    #[inline(always)]
    pub unsafe fn whole(self, head: NonNull<H>) -> NonNull<P> {
        head.cast::<P>()
    }
}

/// A `Large`'s slot begins with its header, which names its vtable.
#[inline(always)]
pub fn slot_header<T>() -> Prefix<Slot<T>, Header> {
    head_at_zero!(Slot<T>, header: Header)
}

/// A closure record begins with its header too.
#[inline(always)]
pub fn record_header<H>() -> Prefix<RecordHead<H>, Header> {
    head_at_zero!(RecordHead<H>, header: Header)
}

// -- A `Large`'s allocation and its vtable (RFC-0102: `Large`, `Record`) ------

/// What every `Large` allocation begins with: the vtable that says what
/// follows and how it is freed.
#[repr(C)]
pub struct Header {
    vtable: &'static Vtable,
}

/// The allocation of a `Large` of one value: `Large::allocate`'s `Box<Slot<T>>`.
#[repr(C)]
pub struct Slot<T> {
    header: Header,
    value: T,
}

impl<T> Slot<T> {
    #[inline(always)]
    pub fn value(&self) -> &T {
        &self.value
    }

    #[inline(always)]
    pub fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }

    #[inline(always)]
    pub fn into_value(self) -> T {
        self.value
    }
}

/// The head of a closure record: its header, its own value, and the length
/// of the tail of elements that follows it from `HeadAndTail`'s `TAIL`.
#[repr(C)]
pub struct RecordHead<H> {
    header: Header,
    value: H,
    len: u16,
}

impl<H> RecordHead<H> {
    #[inline(always)]
    pub fn value(&self) -> &H {
        &self.value
    }

    #[inline(always)]
    pub fn len(&self) -> u16 {
        self.len
    }
}

/// The type a record vtable names: no `Slot<T>` is of it, so a checked read
/// of a slot (`Large::get`) never takes a record for one.
struct RecordOfHeadAndTail<H, E>(PhantomData<fn() -> (H, E)>);

/// How a composite prints inside a `Value`'s `Debug`.
pub trait Show {
    fn show(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// The vtable a `Large` carries in its header. Its fields are private, and
/// a vtable is made only as a `SlotVtable<T>` or a `RecordVtable<H, E>`,
/// whose constructors pair the drop and the print with the allocation they
/// read, so the `TypeId` a vtable states is the allocation its `drop` frees:
/// a `Slot<T>` at `T`'s, a record at its `RecordOfHeadAndTail`'s.
pub struct Vtable {
    type_id: TypeId,
    name: NameFn,
    composite: Option<Composite>,
    drop: unsafe fn(NonNull<Header>),
    debug: Option<unsafe fn(NonNull<Header>, &mut fmt::Formatter<'_>) -> fmt::Result>,
}

impl PartialEq for Vtable {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id
    }
}

impl Vtable {
    #[inline(always)]
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    #[inline(always)]
    pub fn name(&self) -> &'static str {
        (self.name)()
    }

    #[inline(always)]
    pub fn composite(&self) -> Option<Composite> {
        self.composite
    }
}

/// The vtable of a `Slot<T>`: `Large::allocate::<T>` takes one, so the header it
/// writes names the drop and the print of the slot it allocates.
#[repr(transparent)]
pub struct SlotVtable<T>(Vtable, PhantomData<fn() -> T>);

impl<T> SlotVtable<T>
where
    T: 'static,
{
    /// A `Slot<T>` no composite names, printed as its type's name.
    pub const fn drop_only() -> SlotVtable<T> {
        SlotVtable(
            Vtable {
                type_id: TypeId::of::<T>(),
                name: std::any::type_name::<T>,
                composite: None,
                drop: drop_slot::<T>,
                debug: None,
            },
            PhantomData,
        )
    }

    /// A composite's `Slot<T>`, printed as its name.
    pub const fn named(name: NameFn, composite: Composite) -> SlotVtable<T> {
        SlotVtable(
            Vtable {
                type_id: TypeId::of::<T>(),
                name,
                composite: Some(composite),
                drop: drop_slot::<T>,
                debug: None,
            },
            PhantomData,
        )
    }

    /// A composite's `Slot<T>`, printed by `T`'s `Show`.
    pub const fn shown(name: NameFn, composite: Composite) -> SlotVtable<T>
    where
        T: Show,
    {
        SlotVtable(
            Vtable {
                type_id: TypeId::of::<T>(),
                name,
                composite: Some(composite),
                drop: drop_slot::<T>,
                debug: Some(show_slot::<T>),
            },
            PhantomData,
        )
    }

    /// `vtable` as a `Slot<T>`'s, where it is one: a vtable of `T`'s
    /// `TypeId` is made only by a `SlotVtable<T>`, since a record's names a
    /// type private to this module.
    #[inline(always)]
    pub fn of(vtable: &'static Vtable) -> Option<&'static SlotVtable<T>> {
        (vtable.type_id == TypeId::of::<T>()).then(|| {
            // SAFETY: the vtable is a `SlotVtable<T>`'s (above), which is
            // `repr(transparent)` over it.
            unsafe { &*ptr::from_ref(vtable).cast::<SlotVtable<T>>() }
        })
    }

    #[inline(always)]
    pub const fn vtable(&self) -> &Vtable {
        &self.0
    }
}

/// The vtable of a closure record of head `H` and tail elements `E`:
/// `Record::allocate` takes one, as `Large::allocate` takes a `SlotVtable`.
#[repr(transparent)]
pub struct RecordVtable<H, E>(Vtable, PhantomData<fn() -> (H, E)>);

impl<H, E> RecordVtable<H, E>
where
    H: 'static,
    E: 'static,
{
    /// Printed as its name and its tail's length.
    pub const fn new(name: NameFn, composite: Composite) -> RecordVtable<H, E> {
        RecordVtable(
            Vtable {
                type_id: TypeId::of::<RecordOfHeadAndTail<H, E>>(),
                name,
                composite: Some(composite),
                drop: drop_record::<H, E>,
                debug: Some(show_record::<H>),
            },
            PhantomData,
        )
    }

    #[inline(always)]
    pub const fn vtable(&self) -> &Vtable {
        &self.0
    }
}

/// An iterator that yields exactly its `len` elements, which `Record::allocate`
/// writes into a tail of that many without counting them.
///
/// # Safety
/// `next` yields `Some` exactly `len()` times, `len()` read before the first.
pub unsafe trait ExactLen: ExactSizeIterator {}

// SAFETY: a `Map` calls its closure once per element of the slice iterator
// and yields what it returns, and a slice iterator yields its length.
unsafe impl<'a, T, F, B> ExactLen for std::iter::Map<slice::Iter<'a, T>, F> where F: FnMut(&'a T) -> B {}

// SAFETY: `Once` yields its one element, and its `len` is 1 until it does.
unsafe impl<T> ExactLen for std::iter::Once<T> {}

/// # Safety
/// `header` begins a live `Box<Slot<T>>`, freed here and not named after.
unsafe fn drop_slot<T>(header: NonNull<Header>) {
    // SAFETY: the caller's contract.
    drop(unsafe { Box::from_raw(slot_header::<T>().whole(header).as_ptr()) });
}

/// # Safety
/// `header` begins a live `Slot<T>`.
unsafe fn show_slot<T>(header: NonNull<Header>, f: &mut fmt::Formatter<'_>) -> fmt::Result
where
    T: Show,
{
    // SAFETY: the caller's contract.
    unsafe { slot_header::<T>().whole(header).as_ref() }.value.show(f)
}

/// # Safety
/// `header` begins a live record `Record::allocate` wrote at `H` and `E`, freed
/// here and not named after.
unsafe fn drop_record<H, E>(header: NonNull<Header>) {
    // SAFETY: the caller's contract: `Record::allocate` wrote the head and `len`
    // elements after it, in an allocation of `RecordOf::<H, E>::layout(len)`.
    unsafe {
        let head = record_header::<H>().whole(header);
        let len = head.as_ref().len;
        ptr::drop_in_place(RecordOf::<H, E>::tail_run_mut(head, len));
        ptr::drop_in_place(&raw mut (*head.as_ptr()).value);
        RecordOf::<H, E>::dealloc(head, len);
    }
}

/// # Safety
/// `header` begins a live record `Record::allocate` wrote at `H`.
unsafe fn show_record<H>(header: NonNull<Header>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // SAFETY: the caller's contract.
    let head = unsafe { record_header::<H>().whole(header).as_ref() };
    write!(f, "{}({} captures)", head.header.vtable.name(), head.len)
}

type RecordOf<H, E> = HeadAndTail<RecordHead<H>, E>;

/// A `Large`'s allocation, as a `Value` of kind `Large` names it for `'v`.
///
/// `Large::of` trusts two facts of the `Value` it reads, which the code that
/// writes values keeps and no type yet carries: the word of a value of kind
/// `Large` is one `Large::allocate` or `Record::allocate` gave, and the allocation is
/// live until the one `Owned` that holds the value releases it.
///
/// NOTE: the machine keeps both facts: `prepare` gives `Value::inline`
/// inline kinds, and the checker's ownership rules release a value once.
/// Even under `tooling`, code outside the runtime makes no value from bits,
/// builds or edits no `Body`, writes no register, and releases a word only
/// through the holder that owns it, and calls no word as a closure. One gap
/// remains, which the sweep's second part carries in types: a copy of a
/// word read after its holder released it is read as the freed allocation.
#[derive(Clone, Copy)]
pub struct Large<'v>(NonNull<Header>, PhantomData<&'v Value>);

impl Large<'_> {
    /// A `Box<Slot<T>>` holding what `make` returns, as the word a `Value` of
    /// kind `Large` holds. The slot is allocated before `make` runs and its
    /// value is written where the slot lies, so a constructor that reads its
    /// parts inside `make` writes them straight into the heap.
    ///
    /// `make` is the one thing that runs while the slot is uninitialized. If
    /// it unwinds, the slot is still a `Box<MaybeUninit<Slot<T>>>`, whose
    /// drop frees the allocation and runs no `Drop` of `T`.
    ///
    pub fn allocate<T, F>(vtable: &'static SlotVtable<T>, make: F) -> PtrWord
    where
        T: 'static,
        F: FnOnce() -> T,
    {
        let mut slot = Box::<Slot<T>>::new_uninit();
        let at = slot.as_mut_ptr();
        // SAFETY: `at` is the allocation `slot` owns, sized and aligned for a
        // `Slot<T>`. `&raw mut` names each field without reading the
        // uninitialized memory or making a reference to it, and each write
        // puts a valid value in its field.
        unsafe {
            (&raw mut (*at).header).write(Header { vtable: vtable.vtable() });
            (&raw mut (*at).value).write(make());
        }
        // SAFETY: `header` and `value` are both written above, and `Slot<T>`
        // has no other field.
        let slot = unsafe { slot.assume_init() };
        let header = slot_header::<T>().head(NonNull::from(Box::leak(slot)));
        word::word_of_ptr(header.as_ptr())
    }
}

impl<'v> Large<'v> {
    /// The allocation `value` names, where its kind is `Large`.
    #[inline(always)]
    pub fn of(value: &'v Value) -> Option<Large<'v>> {
        (value.kind() == Kind::Large).then(|| {
            let header = word::ptr_of_word::<Header>(PtrWord::from_word(value.word_of_any_kind()));
            // SAFETY: a `Large` word is `Large::allocate`'s or `Record::allocate`'s, the
            // address of a live allocation, which is not null.
            Large(unsafe { NonNull::new_unchecked(header.cast_mut()) }, PhantomData)
        })
    }

    #[inline(always)]
    pub fn vtable(self) -> &'static Vtable {
        // SAFETY: the allocation is live and begins with its header.
        unsafe { self.0.as_ref() }.vtable
    }

    /// Frees the allocation through its vtable.
    #[inline(always)]
    pub fn release(self) {
        // SAFETY: the vtable's `drop` frees the allocation its constructor
        // paired it with, which is this one, and the value that named it is
        // released once.
        unsafe { (self.vtable().drop)(self.0) }
    }

    pub fn fmt(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vtable = self.vtable();
        match vtable.debug {
            // SAFETY: the vtable's print reads the allocation its constructor
            // paired it with, which is this one.
            Some(show) => unsafe { show(self.0, f) },
            None => write!(f, "<{}>", vtable.name()),
        }
    }

    /// The value, where the allocation is a `Slot<T>`.
    #[inline]
    pub fn get<T>(self) -> Option<&'v T>
    where
        T: 'static,
    {
        (self.vtable().type_id == TypeId::of::<T>()).then(|| {
            // SAFETY: the vtable is a `Slot<T>`'s, so the allocation is one,
            // live for `'v`.
            &unsafe { slot_header::<T>().whole(self.0).as_ref() }.value
        })
    }
}

impl Value {
    /// The caller being the value's one holder is not a parameter type yet:
    /// the machine's registers are its callers, and the type that says a
    /// register holds its value alone is the `Operand` of RFC-0102 rule 2.
    /// Until then this is safe and crate-private, as `Value::inline` is, so
    /// no code outside the runtime reaches it.
    #[inline(always)]
    pub(crate) fn release_owned(self) {
        // SAFETY: the caller is the value's one holder: a register the
        // checker's ownership rules release once (RFC-0048 rule 6), or the
        // place a word moved out of one lies in.
        drop(unsafe { Owned::<AcvusRuntime>::from_value(Holding::new(), self) })
    }
}

/// A closure record: a head, then a run of elements, in one allocation.
pub struct Record;

impl Record {
    /// A record of `head` and the elements `tail` yields, as the word a
    /// `Value` of kind `Large` holds.
    ///
    /// # Panics
    /// `tail` holds more than `u16::MAX` elements.
    #[inline(always)]
    pub fn allocate<H, E>(vtable: &'static RecordVtable<H, E>, head: H, tail: &mut dyn ExactLen<Item = E>) -> PtrWord
    where
        H: 'static,
        E: 'static,
    {
        let len = u16::try_from(tail.len()).expect("a closure captures at most u16::MAX registers");
        let at = RecordOf::<H, E>::alloc(len);
        // SAFETY: `at` is `alloc(len)`'s, unwritten. The head is written
        // first, then the `len` elements `tail` yields (`ExactLen`), in
        // order, each below `len` and so inside the allocation.
        unsafe {
            at.as_ptr().write(RecordHead {
                header: Header { vtable: vtable.vtable() },
                value: head,
                len,
            });
            let first = RecordOf::<H, E>::tail(at).as_ptr();
            for (index, element) in tail.enumerate() {
                first.add(index).write(element);
            }
        }
        word::word_of_ptr(record_header::<H>().head(at).as_ptr())
    }
}

// -- A frame's cells as its registers -----------------------------------------

const _: () = assert!(
    mem::offset_of!(Cell, slots) == 0
        && mem::size_of::<Cell>() == CELL_SLOTS as usize * mem::size_of::<Value>()
        && mem::align_of::<Cell>() >= mem::align_of::<Value>(),
    "a run of cells is one run of register slots: each cell is its slots alone, from byte 0"
);
const _: () = assert!(
    mem::size_of::<MaybeUninit<Value>>() == mem::size_of::<Value>()
        && mem::align_of::<MaybeUninit<Value>>() == mem::align_of::<Value>(),
    "a register slot is laid out as the value it holds"
);

/// The first register slot of a run of cells, which a register's `Disp` is
/// a displacement from.
#[inline(always)]
pub fn first_register(cells: NonNull<[Cell]>) -> NonNull<Value> {
    cells.cast::<Value>()
}

/// A run of `len` registers from `at`.
#[derive(Clone, Copy, Debug)]
pub struct Registers {
    pub at: Disp<Value>,
    pub len: u16,
}

/// The registers `run` names in the frame whose cells are `cells`, read for
/// as long as the cells are borrowed.
///
/// # Safety
/// The run lies inside `cells`, and each of its registers holds a value.
#[inline(always)]
pub unsafe fn registers(cells: &[Cell], run: Registers) -> &[Value] {
    // SAFETY: the caller's contract; `first_register` is the cells' first
    // slot, and the borrow of `cells` bounds the result.
    unsafe {
        slice::from_raw_parts(
            at(first_register(NonNull::from(cells)), run.at).as_ptr(),
            usize::from(run.len),
        )
    }
}

/// As `registers`, exclusively.
///
/// # Safety
/// As `registers`'s.
#[inline(always)]
pub unsafe fn registers_mut(cells: &mut [Cell], run: Registers) -> &mut [Value] {
    // SAFETY: as `registers`'s, under the exclusive borrow of `cells`.
    unsafe {
        slice::from_raw_parts_mut(
            at(first_register(NonNull::from(cells)), run.at).as_ptr(),
            usize::from(run.len),
        )
    }
}

/// Two runs of one frame lent at once: one read, one written.
pub struct Apart<'a> {
    pub read: &'a [Value],
    pub written: &'a mut [Value],
}

/// The two runs `read` and `written` name in the frame whose cells are
/// `cells`, lent together for as long as the cells are borrowed.
///
/// # Safety
/// As `registers`'s for each run, and the two runs share no register.
#[inline(always)]
pub unsafe fn registers_apart(cells: &mut [Cell], read: Registers, written: Registers) -> Apart<'_> {
    let first = first_register(NonNull::from(cells));
    // SAFETY: the caller's contract: each run lies inside the cells, the
    // exclusive borrow of which bounds both, and no register of the one is
    // the other's, so the shared and the exclusive name never meet.
    unsafe {
        Apart {
            read: slice::from_raw_parts(at(first, read.at).as_ptr(), usize::from(read.len)),
            written: slice::from_raw_parts_mut(
                at(first, written.at).as_ptr(),
                usize::from(written.len),
            ),
        }
    }
}

/// The first element of a run, where a chain's operand offsets start.
#[inline(always)]
pub fn first_element<T>(run: &[T]) -> NonNull<T> {
    NonNull::from(run).cast::<T>()
}

// -- The running thread's native stack ----------------------------------------

/// `regs::Depth::enter` compares `position` against a `ThreadStack` as a
/// stack that grows down. Linux and Android state a thread's stack; on any
/// other native target the stack is not read yet, and `of_this_thread`
/// states the whole address space, so every frame is admitted and an
/// overflow there aborts (RFC-0100 rule 5).
#[cfg(not(target_arch = "wasm32"))]
pub mod native_stack {

    #[derive(Clone, Copy, Debug)]
    pub struct ThreadStack {
        low: usize,
        high: usize,
    }

    impl ThreadStack {
        pub fn low(self) -> usize {
            self.low
        }

        pub fn high(self) -> usize {
            self.high
        }

        /// Not read on this target: the whole address space, which admits
        /// every frame (the module's head).
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        pub fn of_this_thread() -> Option<ThreadStack> {
            Some(ThreadStack { low: 0, high: usize::MAX })
        }

        /// glibc, musl and bionic state a thread's usable stack through
        /// `pthread_getattr_np`, the main thread's included, with the guard
        /// pages below `low`.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        pub fn of_this_thread() -> Option<ThreadStack> {
            let mut attr = std::mem::MaybeUninit::<libc::pthread_attr_t>::uninit();
            // SAFETY: `pthread_getattr_np` initializes `attr` when it answers
            // 0, and only then is `attr` read and destroyed.
            unsafe {
                if libc::pthread_getattr_np(libc::pthread_self(), attr.as_mut_ptr()) != 0 {
                    return None;
                }
                let mut stack: *mut libc::c_void = std::ptr::null_mut();
                let mut size: libc::size_t = 0;
                let got = libc::pthread_attr_getstack(attr.as_ptr(), &mut stack, &mut size);
                libc::pthread_attr_destroy(attr.as_mut_ptr());
                if got != 0 {
                    return None;
                }
                let low = stack.addr();
                Some(ThreadStack {
                    low,
                    high: low.checked_add(size)?,
                })
            }
        }
    }

    /// The stack pointer, read in place so that no frame is made for a
    /// local to take the address of, and a caller that tail-calls keeps its
    /// tail call.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn position() -> usize {
        let sp: usize;
        // SAFETY: the instruction copies `rsp` into a register and reads or
        // writes nothing else.
        unsafe {
            std::arch::asm!("mov {}, rsp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn position() -> usize {
        let sp: usize;
        // SAFETY: the instruction copies `sp` into a register and reads or
        // writes nothing else.
        unsafe {
            std::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    /// The address of a local of the frame this is inlined into, which is
    /// at or above the stack pointer.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline(always)]
    pub fn position() -> usize {
        let here = 0u8;
        std::ptr::from_ref(&here).addr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_displacement_is_its_slot_times_the_slot_width() {
        let d = Disp::<[u64; 2]>::of(3);
        assert_eq!(d.byte(), 48);
        assert_eq!(d.index(), 3);
        assert_eq!(Disp::<[u64; 2]>::of(4095).byte(), 65520);
        assert!(Disp::<u64>::of(1) < Disp::<u64>::of(2));

        let run = [[1u64, 2], [3, 4], [5, 6]];
        let base = NonNull::from(&run).cast::<[u64; 2]>();
        // SAFETY: `run` has slot 2 and is live while it is read.
        assert_eq!(unsafe { at(base, Disp::of(2)).read() }, [5, 6]);
        // SAFETY: byte 24 is the second word of slot 1, inside `run` and
        // aligned to a `u64`.
        assert_eq!(unsafe { word_at(base, 24).read() }, 4);
    }

    #[test]
    fn a_below_index_and_the_slot_after_it_are_their_slots_times_the_slot_width() {
        let three = Below::<4>::of(3);
        assert_eq!(Disp::<[u64; 2]>::of_below(three), Disp::of(3));
        assert_eq!(Disp::<[u64; 2]>::after(three), Disp::of(4));
        assert_eq!(Disp::<[u64; 2]>::after(Below::<4>::of(0)).byte(), 16);
        let placed = DispBelow::<[u64; 2], 4>::of(three);
        assert_eq!(placed.disp(), Disp::of(3));
        assert_eq!(placed.index(), three);
        assert_eq!(placed.after(), Disp::of(4));
    }

    #[test]
    fn a_below_index_is_checked_when_given_and_composed_without_a_check() {
        assert_eq!(Below::<4>::new(3).map(Below::get), Some(3));
        assert_eq!(Below::<4>::new(4), None);
        assert_eq!(
            Below::<4>::first(3).map(Below::get).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(Below::<64>::masked(63).get(), 63);
        assert_eq!(Below::<64>::masked(64).get(), 0);
        assert_eq!(Below::<64>::masked(u64::MAX.trailing_zeros()).get(), 0);
        assert_eq!(Below::<64>::masked((1u64 << 63).trailing_zeros()).get(), 63);
        let last = Below::<320>::compose(Below::<5>::of(4), Below::<64>::of(63));
        assert_eq!(last.get(), 319);
        assert_eq!(Below::<5>::of(3).step_to(Below::of(4)), Some(Below::of(4)));
        assert_eq!(Below::<5>::of(4).step_to(Below::of(4)), None);
    }

    #[test]
    #[should_panic(expected = "is not below its bound")]
    fn a_below_index_at_its_bound_is_refused() {
        Below::<4>::of(4);
    }

    #[test]
    #[should_panic(expected = "past the bound")]
    fn a_run_of_indices_past_the_bound_is_refused() {
        let _ = Below::<4>::first(5);
    }

    #[test]
    #[should_panic(expected = "does not fit the u16 a Disp holds")]
    fn a_displacement_past_a_u16_is_refused() {
        Disp::<[u64; 2]>::of(4096);
    }

    #[test]
    fn a_tail_starts_at_its_own_alignment() {
        #[repr(C)]
        struct Head {
            _a: u32,
            _b: u32,
            _c: u16,
        }
        assert_eq!(HeadAndTail::<Head, u64>::TAIL, 16);
        assert_eq!(HeadAndTail::<Head, u64>::layout(2).size(), 32);
        assert_eq!(HeadAndTail::<Head, u64>::layout(2).align(), 8);
        assert_eq!(HeadAndTail::<u8, u8>::TAIL, 1);
    }

    #[test]
    fn a_slot_is_named_through_its_header_and_back() {
        static VTABLE: SlotVtable<u64> = SlotVtable::drop_only();
        let prefix = slot_header::<u64>();
        let whole = NonNull::from(Box::leak(Box::new(Slot {
            header: Header { vtable: VTABLE.vtable() },
            value: 7u64,
        })));
        let head = prefix.head(whole);
        assert_eq!(head.addr(), whole.addr());
        // SAFETY: `head` is `whole`'s head, made by `Prefix::head`.
        let back = unsafe { prefix.whole(head) };
        // SAFETY: the slot is live, and freed here once.
        assert_eq!(unsafe { Box::from_raw(back.as_ptr()) }.value, 7);
    }

    #[test]
    fn a_record_holds_its_tail_after_its_head() {
        type Record = HeadAndTail<u32, String>;
        let head = Record::alloc(2);
        // SAFETY: the record is `alloc(2)`'s, written here in full before
        // it is read, and freed once.
        unsafe {
            head.write(2);
            let tail = Record::tail(head).as_ptr();
            tail.write(String::from("a"));
            tail.add(1).write(String::from("b"));
            assert_eq!(Record::tail_run(head, 2), ["a", "b"]);
            Record::tail_run_mut(head, 2)[1].push('c');
            assert_eq!(Record::tail_run(head, 2), ["a", "bc"]);
            std::ptr::drop_in_place(Record::tail_run_mut(head, 2));
            Record::dealloc(head, 2);
        }
    }

    #[test]
    fn a_run_of_registers_is_read_where_its_displacement_names() {
        let mut cells = [Cell::uninit(), Cell::uninit()];
        let values = [5u64, 6, 7, 8].map(|bits| Value::inline(crate::value::Kind::U64, bits));
        let [low, high] = &mut cells;
        for (slot, value) in low.slots[14..].iter_mut().chain(&mut high.slots[..2]).zip(values) {
            slot.write(value);
        }
        let across = Registers {
            at: Disp::of(14),
            len: 4,
        };
        // SAFETY: registers 14..18 are written just above and lie in the two
        // cells.
        let read = unsafe { registers(&cells, across) };
        assert_eq!(read.iter().map(Value::bits).collect::<Vec<_>>(), [5, 6, 7, 8]);
        // SAFETY: as above; 14..16 and 16..18 share no register.
        let apart = unsafe {
            registers_apart(
                &mut cells,
                Registers { at: Disp::of(14), len: 2 },
                Registers { at: Disp::of(16), len: 2 },
            )
        };
        apart.written[0] = apart.read[1];
        // SAFETY: as above.
        let read = unsafe { registers_mut(&mut cells, across) };
        assert_eq!(read.iter().map(Value::bits).collect::<Vec<_>>(), [5, 6, 6, 8]);
        assert_eq!(first_element(&values).addr(), NonNull::from(&values[0]).addr());
    }
}
