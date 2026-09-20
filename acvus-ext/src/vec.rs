//! The functions on `Vec<T>`; the type itself is declared in
//! `acvus_extern`. An element read out of a borrowed container is a
//! reference into it, so the container is neither moved nor changed while
//! the element is in use (RFC-0028).

use acvus_extern::Ctx;
use acvus_extern::{
    Arr, Closure, ClosureFn, Elements, Mut, Ref, Registry, Runtime, Shared, Slice, TransparentOver,
    Var, extern_fn, extern_registry, extern_signature, kind,
};

// A container demotes to a vec (RFC-0027).
extern_signature! {
    ns: "std",
    fn vec<C, T>(items: C) -> Vec<T>
    where
        C: Var<kind::Type>,
        T: Var<kind::Type>;
}

// There is no generic `filled`, and that is a decision. `Runtime` offers
// no clone of a value, so the copies are made in Rust by an instance that
// knows the element type. The language's own clone reaches a `String` as a
// compiler instruction and a `Decimal` as an extension instance, per
// RFC-0020, and neither is a handler a generic body can call.
extern_signature! {
    ns: "vec",
    fn filled<T>(n: u64, x: T) -> Vec<T>
    where
        T: Var<kind::Type>;
}

#[extern_fn(effect = pure)]
fn reverse<T>(mut items: Vec<T>) -> Vec<T>
where
    T: Var<kind::Type>,
{
    items.reverse();
    items
}

#[extern_fn(instance_of = vec, effect = pure)]
#[extern_cast]
fn vec_array<T, N>(items: Arr<T, N>) -> Vec<T>
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
{
    items.0
}

/// A count or a capacity wider than the address space is the overflow
/// `Vec::with_capacity` itself reports.
fn as_len(n: u64) -> usize {
    let Ok(n) = usize::try_from(n) else {
        panic!("capacity overflow")
    };
    n
}

#[extern_fn(effect = pure)]
fn with_capacity<T>(n: u64) -> Vec<T>
where
    T: Var<kind::Type>,
{
    Vec::with_capacity(as_len(n))
}

macro_rules! filled_of {
    ($($name:ident: $t:ty),* $(,)?) => {$(
        #[extern_fn(instance_of = filled, effect = pure)]
        fn $name(n: u64, x: $t) -> Vec<$t> {
            vec![x; as_len(n)]
        }
    )*};
}

filled_of!(
    filled_int: i64,
    filled_float: f64,
    filled_bool: bool,
    filled_str: String,
);

#[extern_fn(effect = pure)]
fn len<T>(c: &Vec<T>) -> u64
where
    T: Var<kind::Type>,
{
    c.len() as u64
}

#[extern_fn(effect = pure)]
fn is_empty<T>(c: &Vec<T>) -> bool
where
    T: Var<kind::Type>,
{
    c.is_empty()
}

/// The whole run of elements, borrowed in place (RFC-0047): the machine
/// indexes this and nothing else. No copy — the slice is a pointer and a
/// length into the container's own storage.
#[extern_fn(effect = pure)]
#[extern_view]
fn as_slice<T, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<T>, Shared, Rt>) -> Slice<T, Shared, Rt>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
#[extern_view]
fn as_slice_mut<T, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<T>, Mut, Rt>) -> Slice<T, Mut, Rt>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
fn first<T, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<T>, Shared, Rt>) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    c.try_map(rt, |c| c.first())
}

#[extern_fn(effect = pure)]
fn last<T, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<T>, Shared, Rt>) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    c.try_map(rt, |c| c.last())
}

#[extern_fn(effect = pure)]
fn push<T>(c: &mut Vec<T>, item: T)
where
    T: Var<kind::Type>,
{
    c.push(item);
}

#[extern_fn(effect = pure)]
fn pop<T>(c: &mut Vec<T>) -> Option<T>
where
    T: Var<kind::Type>,
{
    c.pop()
}

#[extern_fn(effect = pure)]
fn insert<T>(c: &mut Vec<T>, index: u64, item: T)
where
    T: Var<kind::Type>,
{
    let Ok(index) = usize::try_from(index) else {
        panic!(
            "insertion index (is {index}) should be <= len (is {})",
            c.len()
        )
    };
    c.insert(index, item);
}

#[extern_fn(effect = pure)]
fn remove<T>(c: &mut Vec<T>, index: u64) -> T
where
    T: Var<kind::Type>,
{
    let Ok(index) = usize::try_from(index) else {
        panic!(
            "removal index (is {index}) should be < len (is {})",
            c.len()
        )
    };
    c.remove(index)
}

#[extern_fn(effect = pure)]
fn clear<T>(c: &mut Vec<T>)
where
    T: Var<kind::Type>,
{
    c.clear();
}

#[extern_fn(effect = pure)]
fn truncate<T>(c: &mut Vec<T>, len: u64)
where
    T: Var<kind::Type>,
{
    c.truncate(usize::try_from(len).unwrap_or(usize::MAX));
}

#[extern_fn(effect = pure)]
fn extend<T>(c: &mut Vec<T>, items: Vec<T>)
where
    T: Var<kind::Type>,
{
    c.extend(items);
}

/// An index wider than the address space is past every length, and the
/// message is the one `slice::swap` panics with at an index it can hold.
fn swap_index(index: u64, len: usize) -> usize {
    let Ok(index) = usize::try_from(index) else {
        panic!("index out of bounds: the len is {len} but the index is {index}")
    };
    index
}

#[extern_fn(effect = pure)]
fn swap<T>(c: &mut Vec<T>, i: u64, j: u64)
where
    T: Var<kind::Type>,
{
    let len = c.len();
    c.swap(swap_index(i, len), swap_index(j, len));
}

// -- ordered by the element ---------------------------------------------

// A container of `Erased<Rt, T>` still has no crossing that reads its
// storage, and a concrete `&Vec<i64>` parameter is not the way around it:
// the storage is a `Vec<Owned<Rt>>`, so reading it back as a `Vec<i64>`
// traps at run time with `NO_VEC_STORAGE`. Each of these is therefore a
// shared signature (RFC-0019) with one instance per concrete element type.
//
// `min` and `max` are not here either, and that is a decision. They are
// `iter::`'s, and declaring them for `vec` as well makes the bare name
// unresolvable: `as_iter(&scores) | max` stops compiling, refused as
// "declared by iter::max and vec::max". A vec's least element is
// `into_iter(v) | min`; the aggregates take the element by value, so the
// borrowing `as_iter(&v) | min` is refused for the element's type.
//
// `dedup`, `fill` and `resize` are not among them, and that is a bound
// rather than an omission: the first two change the length and the third
// overwrites an owned value, and a run of the container's values is of
// fixed length and holds elements no handler may drop. A script dedups
// with `into_iter(v) | dedup | collect` and fills with `filled(n, x)`.

/// Rust's `f64` has no `Ord`, and `total_cmp` is the order its own float
/// slices sort by.
pub(crate) trait Order: PartialEq + Clone + Send + Sync + 'static {
    fn order(&self, other: &Self) -> std::cmp::Ordering;
}

macro_rules! ord_by_cmp {
    ($($t:ty),* $(,)?) => {$(
        impl Order for $t {
            fn order(&self, other: &Self) -> std::cmp::Ordering {
                Ord::cmp(self, other)
            }
        }
    )*};
}

ord_by_cmp!(i64, u64, bool, String);

impl Order for f64 {
    fn order(&self, other: &Self) -> std::cmp::Ordering {
        f64::total_cmp(self, other)
    }
}

/// The element at `at`, at the type the container's values were erased
/// from (RFC-0047 rule 1).
///
/// # Safety
/// `at` is below the run's length, the container is live for the call
/// (RFC-0018), and every value in the run was erased from `T`.
pub(crate) unsafe fn element<'a, T, Rt>(rt: &'a Rt, run: &Elements<Rt>, at: usize) -> &'a T
where
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    // SAFETY: the caller's contract, forwarded.
    unsafe { rt.value_as_ref::<T>(run.at(at)) }
}

/// # Safety
/// As `element`, for every index of the run.
pub(crate) unsafe fn elements<'a, T, Rt>(
    rt: &'a Rt,
    run: &'a Elements<Rt>,
) -> impl Iterator<Item = &'a T>
where
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    // SAFETY: the caller's contract, forwarded; `at` comes from the run's
    // own length.
    (0..run.len()).map(move |at| unsafe { element::<T, Rt>(rt, run, at) })
}

/// # Safety
/// As `elements`.
pub(crate) unsafe fn is_ordered<T, Rt>(rt: &Rt, run: &Elements<Rt>) -> bool
where
    T: Order,
    Rt: Runtime,
{
    // SAFETY: the caller's contract, forwarded.
    let mut seen = unsafe { elements::<T, Rt>(rt, run) };
    let Some(mut previous) = seen.next() else {
        return true;
    };
    seen.all(|next| {
        let ordered = previous.order(next) != std::cmp::Ordering::Greater;
        previous = next;
        ordered
    })
}

/// # Safety
/// As `elements`.
pub(crate) unsafe fn found<T, Rt>(rt: &Rt, run: &Elements<Rt>, x: &T) -> Option<u64>
where
    T: Order,
    Rt: Runtime,
{
    let (mut lo, mut hi) = (0usize, run.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        // SAFETY: the caller's contract, forwarded; `mid` is below `hi`,
        // which is at most the run's length.
        match unsafe { element::<T, Rt>(rt, run, mid) }.order(x) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(mid as u64),
        }
    }
    None
}

/// # Safety
/// As `elements`, and `run` is an exclusive take of its container
/// (RFC-0047 §2).
unsafe fn sort_run<T, Rt>(rt: &Rt, run: &Elements<Rt>)
where
    T: Order,
    Rt: Runtime,
{
    let n = run.len();
    let mut order: Vec<usize> = (0..n).collect();
    // SAFETY: the caller's contract, forwarded.
    order.sort_by(|&a, &b| unsafe {
        element::<T, Rt>(rt, run, a).order(element::<T, Rt>(rt, run, b))
    });
    // SAFETY: `order` is a permutation of `0..n`, so its inverse is one
    // too, and the caller's contract gives the run exclusively.
    unsafe { permute(run, destinations_of(&order)) };
}

extern_signature! {
    ns: "vec",
    fn sort<T>(c: &mut Vec<T>)
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "vec",
    fn contains<T>(c: &Vec<T>, x: &T) -> bool
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "vec",
    fn binary_search<T>(c: &Vec<T>, x: &T) -> Option<u64>
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "vec",
    fn is_sorted<T>(c: &Vec<T>) -> bool
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "vec",
    fn to_vec<T>(c: &Vec<T>) -> Vec<T>
    where
        T: Var<kind::Type>;
}

/// # Safety
/// As `elements`, for both runs, and `offset + part.len() <= whole.len()`.
unsafe fn run_matches_at<T, Rt>(
    rt: &Rt,
    whole: &Elements<Rt>,
    part: &Elements<Rt>,
    offset: usize,
) -> bool
where
    T: Order,
    Rt: Runtime,
{
    (0..part.len()).all(|k| {
        // SAFETY: the caller's contract, forwarded.
        unsafe { element::<T, Rt>(rt, whole, offset + k) == element::<T, Rt>(rt, part, k) }
    })
}

fn repeat_len(len: usize, times: usize) -> usize {
    let Some(total) = len.checked_mul(times) else {
        panic!("capacity overflow")
    };
    total
}

type KeyOf<T, E, Rt> = Closure<(Ref<T, Shared, Rt>,), i64, E, Rt>;

fn keyed_order(keys: Vec<i64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&at| keys[at]);
    order
}

fn sort_by_key_now<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<T>, Mut, Rt>, f: KeyOf<T, E, Rt>)
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let run = Slice::<T, Mut, Rt>::of(c.elements(rt)).into_elements();
    let keys: Vec<i64> = (0..run.len())
        .map(|at| {
            // SAFETY: `at` is below the run's length and the container is
            // live for the call (RFC-0018); the closure only reads.
            let element = unsafe { Ref::lend(rt, run.at(at)) };
            f.call_now(ctx, (element,))
        })
        .collect();
    // SAFETY: `keyed_order` permutes `0..len`, so its inverse is one too,
    // and a `Mut` reference takes the run exclusively (RFC-0047 §2).
    unsafe { permute(&run, destinations_of(&keyed_order(keys))) };
}

#[extern_fn(effect = E, sync = sort_by_key_now)]
async fn sort_by_key<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<T>, Mut, Rt>, f: KeyOf<T, E, Rt>)
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let run = Slice::<T, Mut, Rt>::of(c.elements(rt)).into_elements();
    let mut keys: Vec<i64> = Vec::with_capacity(run.len());
    for at in 0..run.len() {
        // SAFETY: as in `sort_by_key_now`.
        let element = unsafe { Ref::lend(rt, run.at(at)) };
        keys.push(f.call(ctx, (element,)).await);
    }
    // SAFETY: as in `sort_by_key_now`.
    unsafe { permute(&run, destinations_of(&keyed_order(keys))) };
}

extern_signature! {
    ns: "vec",
    fn starts_with<T>(c: &Vec<T>, prefix: &Vec<T>) -> bool
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "vec",
    fn ends_with<T>(c: &Vec<T>, suffix: &Vec<T>) -> bool
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "vec",
    fn repeat<T>(c: &Vec<T>, times: u64) -> Vec<T>
    where
        T: Var<kind::Type>;
}

macro_rules! ordered_of {
    (
        $t:ty,
        sort: $sort:ident,
        contains: $contains:ident,
        binary_search: $binary_search:ident,
        is_sorted: $is_sorted:ident,
        to_vec: $to_vec:ident,
        starts_with: $starts_with:ident,
        ends_with: $ends_with:ident,
        repeat: $repeat:ident,
    ) => {
        #[extern_fn(instance_of = sort, effect = pure)]
        fn $sort<Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<$t>, Mut, Rt>)
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let run = Slice::<$t, Mut, Rt>::of(c.elements(rt)).into_elements();
            // SAFETY: the run is this container's own values, every one
            // erased from `$t`, and a `Mut` reference takes it exclusively.
            unsafe { sort_run::<$t, Rt>(rt, &run) };
        }

        #[extern_fn(instance_of = contains, effect = pure)]
        fn $contains<Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<$t>, Shared, Rt>, x: &$t) -> bool
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let run = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            // SAFETY: as `$sort`, shared.
            unsafe { elements::<$t, Rt>(rt, &run) }.any(|element| element == x)
        }

        #[extern_fn(instance_of = binary_search, effect = pure)]
        fn $binary_search<Rt>(
            ctx: &mut Ctx<'_, Rt>,
            c: Ref<Vec<$t>, Shared, Rt>,
            x: &$t,
        ) -> Option<u64>
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let run = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            // SAFETY: as `$sort`, shared.
            unsafe { found::<$t, Rt>(rt, &run, x) }
        }

        #[extern_fn(instance_of = is_sorted, effect = pure)]
        fn $is_sorted<Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<$t>, Shared, Rt>) -> bool
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let run = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            // SAFETY: as `$sort`, shared.
            unsafe { is_ordered::<$t, Rt>(rt, &run) }
        }

        #[extern_fn(instance_of = to_vec, effect = pure)]
        fn $to_vec<Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<$t>, Shared, Rt>) -> Vec<$t>
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let run = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            // SAFETY: as `$sort`, shared.
            unsafe { elements::<$t, Rt>(rt, &run) }.cloned().collect()
        }

        #[extern_fn(instance_of = starts_with, effect = pure)]
        fn $starts_with<Rt>(
            ctx: &mut Ctx<'_, Rt>,
            c: Ref<Vec<$t>, Shared, Rt>,
            prefix: Ref<Vec<$t>, Shared, Rt>,
        ) -> bool
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let whole = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            let part = Slice::<$t, Shared, Rt>::of(prefix.elements(rt)).into_elements();
            // SAFETY: as `$sort`, shared, and the length guard is the
            // offset contract.
            part.len() <= whole.len() && unsafe { run_matches_at::<$t, Rt>(rt, &whole, &part, 0) }
        }

        #[extern_fn(instance_of = ends_with, effect = pure)]
        fn $ends_with<Rt>(
            ctx: &mut Ctx<'_, Rt>,
            c: Ref<Vec<$t>, Shared, Rt>,
            suffix: Ref<Vec<$t>, Shared, Rt>,
        ) -> bool
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let whole = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            let part = Slice::<$t, Shared, Rt>::of(suffix.elements(rt)).into_elements();
            let Some(offset) = whole.len().checked_sub(part.len()) else {
                return false;
            };
            // SAFETY: as `$starts_with`; `checked_sub` is the offset
            // contract.
            unsafe { run_matches_at::<$t, Rt>(rt, &whole, &part, offset) }
        }

        #[extern_fn(instance_of = repeat, effect = pure)]
        fn $repeat<Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<$t>, Shared, Rt>, times: u64) -> Vec<$t>
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let run = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            let times = as_len(times);
            let mut out: Vec<$t> = Vec::with_capacity(repeat_len(run.len(), times));
            for _ in 0..times {
                // SAFETY: as `$sort`, shared.
                out.extend(unsafe { elements::<$t, Rt>(rt, &run) }.cloned());
            }
            out
        }
    };
}

ordered_of!(
    i64,
    sort: sort_int,
    contains: contains_int,
    binary_search: binary_search_int,
    is_sorted: is_sorted_int,
    to_vec: to_vec_int,
    starts_with: starts_with_int,
    ends_with: ends_with_int,
    repeat: repeat_int,
);

ordered_of!(
    u64,
    sort: sort_index,
    contains: contains_index,
    binary_search: binary_search_index,
    is_sorted: is_sorted_index,
    to_vec: to_vec_index,
    starts_with: starts_with_index,
    ends_with: ends_with_index,
    repeat: repeat_index,
);

ordered_of!(
    f64,
    sort: sort_float,
    contains: contains_float,
    binary_search: binary_search_float,
    is_sorted: is_sorted_float,
    to_vec: to_vec_float,
    starts_with: starts_with_float,
    ends_with: ends_with_float,
    repeat: repeat_float,
);

ordered_of!(
    bool,
    sort: sort_bool,
    contains: contains_bool,
    binary_search: binary_search_bool,
    is_sorted: is_sorted_bool,
    to_vec: to_vec_bool,
    starts_with: starts_with_bool,
    ends_with: ends_with_bool,
    repeat: repeat_bool,
);

ordered_of!(
    String,
    sort: sort_str,
    contains: contains_str,
    binary_search: binary_search_str,
    is_sorted: is_sorted_str,
    to_vec: to_vec_str,
    starts_with: starts_with_str,
    ends_with: ends_with_str,
    repeat: repeat_str,
);

// -- the vec's own ------------------------------------------------------

#[extern_fn(effect = pure)]
fn capacity<T>(c: &Vec<T>) -> u64
where
    T: Var<kind::Type>,
{
    c.capacity() as u64
}

#[extern_fn(effect = pure)]
fn shrink_to_fit<T>(c: &mut Vec<T>)
where
    T: Var<kind::Type>,
{
    c.shrink_to_fit();
}

#[extern_fn(effect = pure)]
fn split_off<T>(c: &mut Vec<T>, at: u64) -> Vec<T>
where
    T: Var<kind::Type>,
{
    let Ok(at) = usize::try_from(at) else {
        panic!(
            "`at` split index (is {at}) should be <= len (is {})",
            c.len()
        )
    };
    c.split_off(at)
}

// There is no `get_mut`, `first_mut` or `last_mut`. Each would answer
// `Option<Ref<T, Mut, Rt>>`, and the language does not carry a `&mut`
// inside an `Option`: a store through what such a call binds is refused
// with "cannot store through &_: not a `&mut`". A script writes to an
// element through `v[i] = x`, which is the indexing instruction and keeps
// the exclusive loan at the place (RFC-0047).

#[extern_fn(effect = pure)]
fn get<T, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    c: Ref<Vec<T>, Shared, Rt>,
    at: u64,
) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let at = usize::try_from(at).ok()?;
    c.try_map(rt, |c| c.get(at))
}

// -- the view -----------------------------------------------------------

/// The `&[T]` surface, Rust's `impl<T> [T]`, reached from a `Vec` and from
/// an array alike through `as_slice`. A handler here sees a run of the
/// runtime's own values and no element type (RFC-0047 §1), so an operation
/// that only moves elements is generic in `T` and one that reads an
/// element is a shared signature with an instance per element type.
#[extern_fn(name = "len", effect = pure)]
fn slice_len<T, Rt>(s: Slice<T, Shared, Rt>) -> u64
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    s.into_elements().len() as u64
}

// There is no `slice::reverse`, and that is a collision rather than a
// bound. `vec::reverse` already holds the bare name with a by-value
// `Vec<T> -> Vec<T>` signature, and adding a second declaration stops a
// call that used to settle: `reverse(v)` on a `Vec<#Float>` is refused
// with "no `reverse` takes a call of type Fn(Vec<#Float>) -> Vec<_>". A
// view is reversed by `swap` over its halves until either `vec::reverse`
// takes Rust's in-place shape or RFC-0043 settles the pair.

#[extern_fn(name = "is_empty", effect = pure)]
fn slice_is_empty<T, Rt>(s: Slice<T, Shared, Rt>) -> bool
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    s.into_elements().is_empty()
}

/// `slice::swap` bounds-checks before it exchanges; this view's `at_mut`
/// does not, so the check is here, with Rust's own message.
fn within(index: u64, len: usize) -> usize {
    let index = swap_index(index, len);
    if index >= len {
        panic!("index out of bounds: the len is {len} but the index is {index}")
    }
    index
}

#[extern_fn(name = "swap", effect = pure)]
fn slice_swap<T, Rt>(s: Slice<T, Mut, Rt>, i: u64, j: u64)
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let s = s.into_elements();
    let len = s.len();
    let (i, j) = (within(i, len), within(j, len));
    // SAFETY: `within` put both indices below the length, and a `Mut` slice
    // is an exclusive take of its container (RFC-0047 §2).
    unsafe { std::ptr::swap(s.at_mut(i), s.at_mut(j)) };
}

#[extern_fn(name = "get", effect = pure)]
fn slice_get<T, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    s: Slice<T, Shared, Rt>,
    at: u64,
) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let s = s.into_elements();
    let at = usize::try_from(at).ok()?;
    (at < s.len()).then(|| {
        // SAFETY: `at` is below the length, and the container the caller
        // lent is live for the call (RFC-0018).
        Ref::lend(rt, unsafe { s.at(at) })
    })
}

#[extern_fn(name = "first", effect = pure)]
fn slice_first<T, Rt>(ctx: &mut Ctx<'_, Rt>, s: Slice<T, Shared, Rt>) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let s = s.into_elements();
    (!s.is_empty()).then(|| {
        // SAFETY: the run is non-empty, and the container the caller lent
        // is live for the call (RFC-0018).
        Ref::lend(rt, unsafe { s.at(0) })
    })
}

#[extern_fn(name = "last", effect = pure)]
fn slice_last<T, Rt>(ctx: &mut Ctx<'_, Rt>, s: Slice<T, Shared, Rt>) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let s = s.into_elements();
    let last = s.len().checked_sub(1)?;
    // SAFETY: `last` is the final index of a non-empty run, and the
    // container the caller lent is live for the call (RFC-0018).
    Some(Ref::lend(rt, unsafe { s.at(last) }))
}

/// Where each element goes, read off the order it came out of the sort in:
/// the element now at `sources[k]` belongs at `k`.
fn destinations_of(sources: &[usize]) -> Vec<usize> {
    let mut destinations = vec![0usize; sources.len()];
    for (at, &source) in sources.iter().enumerate() {
        destinations[source] = at;
    }
    destinations
}

/// # Safety
/// `destinations` is a permutation of `0..s.len()`, and `s` is an exclusive
/// take of its container (RFC-0047 §2).
unsafe fn permute<Rt>(s: &Elements<Rt>, mut destinations: Vec<usize>)
where
    Rt: Runtime,
{
    for at in 0..destinations.len() {
        while destinations[at] != at {
            let to = destinations[at];
            // SAFETY: the caller's contract: both are indices of the run,
            // exclusively named.
            unsafe { std::ptr::swap(s.at_mut(at), s.at_mut(to)) };
            destinations.swap(at, to);
        }
    }
}

#[derive(Clone, Copy)]
struct MergeRun {
    lo: usize,
    mid: usize,
    hi: usize,
}

fn merge_runs(n: usize, width: usize) -> impl Iterator<Item = MergeRun> {
    (0..n).step_by(width * 2).map(move |lo| MergeRun {
        lo,
        mid: (lo + width).min(n),
        hi: (lo + width * 2).min(n),
    })
}

/// The comparator answers −1/0/1, the protocol `string::cmp` already
/// speaks; the language has no `Ordering` type for it to return.
fn takes_left(verdict: i64) -> bool {
    verdict <= 0
}

type Comparator<T, E, Rt> = Closure<(Ref<T, Shared, Rt>, Ref<T, Shared, Rt>), i64, E, Rt>;

fn sort_by_now<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<T>, Mut, Rt>, f: Comparator<T, E, Rt>)
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let s = Slice::<T, Mut, Rt>::of(c.elements(rt)).into_elements();
    let n = s.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut merged: Vec<usize> = Vec::with_capacity(n);
    let mut width = 1usize;
    while width < n {
        for run in merge_runs(n, width) {
            let mut left = run.lo;
            let mut right = run.mid;
            merged.clear();
            while left < run.mid && right < run.hi {
                // SAFETY: both are indices of the run, and the container the
                // caller lent is live for the call (RFC-0018). The
                // comparator only reads, so no write to the run is live
                // while the two references are.
                let a = unsafe { Ref::lend(rt, s.at(order[left])) };
                let b = unsafe { Ref::lend(rt, s.at(order[right])) };
                let taken = if takes_left(f.call_now(ctx, (a, b))) {
                    let at = order[left];
                    left += 1;
                    at
                } else {
                    let at = order[right];
                    right += 1;
                    at
                };
                merged.push(taken);
            }
            merged.extend_from_slice(&order[left..run.mid]);
            merged.extend_from_slice(&order[right..run.hi]);
            order[run.lo..run.hi].copy_from_slice(&merged);
        }
        width *= 2;
    }
    // SAFETY: `order` is a permutation of `0..n` — it starts as one and the
    // merges only reorder it — so its inverse is one too, and a `Mut` slice
    // is an exclusive take of its container (RFC-0047 §2).
    unsafe { permute(&s, destinations_of(&order)) };
}

#[extern_fn(effect = E, sync = sort_by_now)]
async fn sort_by<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Vec<T>, Mut, Rt>, f: Comparator<T, E, Rt>)
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let s = Slice::<T, Mut, Rt>::of(c.elements(rt)).into_elements();
    let n = s.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut merged: Vec<usize> = Vec::with_capacity(n);
    let mut width = 1usize;
    while width < n {
        for run in merge_runs(n, width) {
            let mut left = run.lo;
            let mut right = run.mid;
            merged.clear();
            while left < run.mid && right < run.hi {
                // SAFETY: as in `sort_by_now`.
                let a = unsafe { Ref::lend(rt, s.at(order[left])) };
                let b = unsafe { Ref::lend(rt, s.at(order[right])) };
                let taken = if takes_left(f.call(ctx, (a, b)).await) {
                    let at = order[left];
                    left += 1;
                    at
                } else {
                    let at = order[right];
                    right += 1;
                    at
                };
                merged.push(taken);
            }
            merged.extend_from_slice(&order[left..run.mid]);
            merged.extend_from_slice(&order[right..run.hi]);
            order[run.lo..run.hi].copy_from_slice(&merged);
        }
        width *= 2;
    }
    // SAFETY: as in `sort_by_now`.
    unsafe { permute(&s, destinations_of(&order)) };
}

fn rotated(len: usize, mid: usize) -> Vec<usize> {
    (0..len).map(|at| (at + mid) % len).collect()
}

/// Rust's `rotate_left` panics unless `mid <= len`, and this view's
/// `at_mut` bounds-checks only under `debug_assert!`, so the check is here.
fn rotation_index(mid: u64, len: usize) -> usize {
    let Ok(mid) = usize::try_from(mid) else {
        panic!("rotation index (is {mid}) should be <= len (is {len})")
    };
    if mid > len {
        panic!("rotation index (is {mid}) should be <= len (is {len})")
    }
    mid
}

#[extern_fn(name = "rotate_left", effect = pure)]
fn slice_rotate_left<T, Rt>(s: Slice<T, Mut, Rt>, mid: u64)
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let s = s.into_elements();
    let mid = rotation_index(mid, s.len());
    // SAFETY: `rotated` is a permutation of `0..len`, so its inverse is one
    // too, and a `Mut` slice is an exclusive take of its container
    // (RFC-0047 §2).
    unsafe { permute(&s, destinations_of(&rotated(s.len(), mid))) };
}

#[extern_fn(name = "rotate_right", effect = pure)]
fn slice_rotate_right<T, Rt>(s: Slice<T, Mut, Rt>, k: u64)
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let s = s.into_elements();
    let len = s.len();
    let k = rotation_index(k, len);
    // SAFETY: as `slice_rotate_left`; rotating right by `k` is rotating
    // left by `len - k`.
    unsafe { permute(&s, destinations_of(&rotated(len, len - k))) };
}

pub fn slice_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "slice",
        fns: [
            slice_len, slice_is_empty, slice_swap,
            slice_get, slice_first, slice_last,
            slice_rotate_left, slice_rotate_right,
        ],
    }
}

pub fn vec_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "vec",
        types: [Vec<_>],
        signatures: [
            vec, filled, sort, contains, binary_search, is_sorted, to_vec,
            starts_with, ends_with, repeat,
        ],
        fns: [
            reverse, vec_array, with_capacity,
            filled_int, filled_float, filled_bool, filled_str,
            len, is_empty, as_slice, as_slice_mut, first, last,
            push, pop, insert, remove, clear, truncate, extend, swap,
            sort_by, sort_by_key, capacity, shrink_to_fit, split_off,
            get,
            sort_int, contains_int, binary_search_int, is_sorted_int,
            to_vec_int, starts_with_int, ends_with_int, repeat_int,
            sort_index, contains_index, binary_search_index, is_sorted_index,
            to_vec_index, starts_with_index, ends_with_index, repeat_index,
            sort_float, contains_float, binary_search_float, is_sorted_float,
            to_vec_float, starts_with_float, ends_with_float, repeat_float,
            sort_bool, contains_bool, binary_search_bool, is_sorted_bool,
            to_vec_bool, starts_with_bool, ends_with_bool, repeat_bool,
            sort_str, contains_str, binary_search_str, is_sorted_str,
            to_vec_str, starts_with_str, ends_with_str, repeat_str,
        ],
    }
}
