//! The `&[T]` surface, Rust's `impl<T> [T]`: every operation over a
//! container's *elements* lives here once, and a `Vec<T>` or an
//! `Array<T, N>` reaches it through its own `as_slice` view (RFC-0047 rule 3),
//! the way Rust's `[T]` serves `Vec<T>` and `[T; N]` through deref.
//!
//! A handler takes Rust's own `&[T]` / `&mut [T]` at the run's lifetime
//! (RFC-0068 rule 4): the crossing made the borrow, so nothing here reads a
//! value back at a type the acvus checker did not settle.
//!
//! An operation that only moves elements is generic in `T`. One that *reads*
//! an element is a shared signature (RFC-0019) with one instance per
//! concrete element type, and an instance names its element as
//! `Erased<Rt, $t>`: the acvus type is the element's own (`Erased<R, T>`'s
//! `poly_ty` is `T`'s), and the element is read in place through
//! `Erased::as_ref`, the crossing's own read of a value it erased.

use acvus_extern::{
    Ctx, Erased, Registry, Runtime, Stored, TransparentOver, Var, extern_fn, extern_registry,
    extern_signature, kind,
};

use crate::vec::as_len;

// -- indices ------------------------------------------------------------

/// An index wider than the address space is past every length, and the
/// message is the one `slice::swap` panics with at an index it can hold.
pub(crate) fn swap_index(index: u64, len: usize) -> usize {
    let Ok(index) = usize::try_from(index) else {
        panic!("index out of bounds: the len is {len} but the index is {index}")
    };
    index
}

/// Rust's `rotate_left` panics unless `mid <= len`, and the check is here
/// so an index wider than the address space gives that same message.
fn rotation_index(mid: u64, len: usize) -> usize {
    let Ok(mid) = usize::try_from(mid) else {
        panic!("rotation index (is {mid}) should be <= len (is {len})")
    };
    if mid > len {
        panic!("rotation index (is {mid}) should be <= len (is {len})")
    }
    mid
}

/// Moves the element at `sources[k]` to `k`, by Rust's own swaps: the order
/// a sort produced, applied to the run it was produced from.
pub(crate) fn permute<T>(xs: &mut [T], sources: &[usize]) {
    let mut destinations = vec![0usize; sources.len()];
    for (at, &source) in sources.iter().enumerate() {
        destinations[source] = at;
    }
    for at in 0..destinations.len() {
        while destinations[at] != at {
            let to = destinations[at];
            xs.swap(at, to);
            destinations.swap(at, to);
        }
    }
}

// -- the moving half, generic in the element ----------------------------

/// `total`: `len` reads the length the slice holds, and widening a `usize` to
/// `u64` neither fails nor panics on any target Rust supports.
#[extern_fn(effect = pure, total, ensures(ret = len(s)))]
fn len<T, Rt>(s: &[T]) -> u64
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    s.len() as u64
}

// There is no `slice::reverse`, and that is a collision rather than a
// bound. `vec::reverse` holds the bare name with a by-value
// `Vec<T> -> Vec<T>` signature, and a second declaration stops a call that
// used to settle: `reverse(x)` over a `Vec<#Float>` is refused with "no
// `reverse` takes a call of type Fn(Vec<#Float>) -> Vec<_>". Measured one
// variable apart — registering it fails
// `acvus-mir-test/tests/regression_0041.rs`'s
// `a_specialized_local_pays_one_erase_per_generic_consumer_and_none_at_its_member`,
// unregistering it passes. A view is reversed by `swap` over its halves
// until either `vec::reverse` takes Rust's in-place shape or RFC-0043
// settles the pair.

#[extern_fn(effect = pure)]
fn is_empty<T, Rt>(s: &[T]) -> bool
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    s.is_empty()
}

#[extern_fn(effect = pure)]
fn swap<T, Rt>(s: &mut [T], i: u64, j: u64)
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    let len = s.len();
    let i = swap_index(i, len);
    let j = swap_index(j, len);
    s.swap(i, j);
}

// There is no `get_mut`, `first_mut` or `last_mut`. Each would answer
// `Option<&mut T>`, and the language does not carry a `&mut` inside an
// `Option`: a store through what such a call binds is refused with "cannot
// store through &_: not a `&mut`". A script writes to an element through
// `v[i] = x`, which is the indexing instruction and keeps the exclusive
// loan at the place (RFC-0047).

#[extern_fn(effect = pure)]
fn get<T, Rt>(s: &[T], at: u64) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    s.get(usize::try_from(at).ok()?)
}

#[extern_fn(effect = pure)]
fn first<T, Rt>(s: &[T]) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    s.first()
}

#[extern_fn(effect = pure)]
fn last<T, Rt>(s: &[T]) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    s.last()
}

#[extern_fn(effect = pure)]
fn rotate_left<T, Rt>(s: &mut [T], mid: u64)
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    let mid = rotation_index(mid, s.len());
    s.rotate_left(mid);
}

#[extern_fn(effect = pure)]
fn rotate_right<T, Rt>(s: &mut [T], k: u64)
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    let k = rotation_index(k, s.len());
    s.rotate_right(k);
}

// -- ordered by the element ---------------------------------------------

// Each of these reads an element at its own type, so each is a shared
// signature (RFC-0019) with one instance per concrete element type. An
// instance names its element as `Erased<Rt, $t>`: the acvus type is the
// element's own, and the element is read in place through `Erased::as_ref`,
// the crossing's own read of a value it erased.
//
// `dedup`, `fill` and `resize` are not among them, and that is a bound
// rather than an omission: the first two change the length and the third
// overwrites an owned value, and a view is of fixed length and holds
// elements no handler may drop. A script dedups with
// `into_iter(v) | dedup | collect` and fills with `filled(n, x)`.

/// Rust's `f64` has no `Ord`, and `total_cmp` is the order its own float
/// slices sort by.
trait Order: PartialEq + Clone + Send + Sync + 'static {
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

/// The run's elements, each at the type it was erased from (RFC-0047
/// rule 1). `Erased::as_ref` is the crossing's own read, so no element is
/// read back at a type the acvus checker did not settle.
fn elements<'a, T, Rt>(rt: &'a Rt, s: &'a [Erased<Rt, T>]) -> impl Iterator<Item = &'a T> + 'a
where
    T: Order + Stored<Rt>,
    Rt: Runtime,
{
    s.iter().map(move |cell| cell.as_ref(rt))
}

fn is_ordered<T, Rt>(rt: &Rt, s: &[Erased<Rt, T>]) -> bool
where
    T: Order + Stored<Rt>,
    Rt: Runtime,
{
    s.windows(2)
        .all(|pair| pair[0].as_ref(rt).order(pair[1].as_ref(rt)) != std::cmp::Ordering::Greater)
}

/// Rust's `binary_search` carries where a miss would go in its `Err`; the
/// language has no `Result<u64, u64>` to tell two indices of one meaning
/// apart, so a miss is `None`.
fn found<T, Rt>(rt: &Rt, s: &[Erased<Rt, T>], x: &T) -> Option<u64>
where
    T: Order + Stored<Rt>,
    Rt: Runtime,
{
    let mut lo = 0usize;
    let mut hi = s.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match s[mid].as_ref(rt).order(x) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(mid as u64),
        }
    }
    None
}

/// The first of equal elements, as Rust's `Iterator::min`.
fn least<T, Rt>(rt: &Rt, s: &[Erased<Rt, T>]) -> Option<T>
where
    T: Order + Stored<Rt>,
    Rt: Runtime,
{
    let mut seen = elements(rt, s);
    let mut best = seen.next()?;
    for next in seen {
        if next.order(best) == std::cmp::Ordering::Less {
            best = next;
        }
    }
    Some(best.clone())
}

/// The last of equal elements, as Rust's `Iterator::max`.
fn greatest<T, Rt>(rt: &Rt, s: &[Erased<Rt, T>]) -> Option<T>
where
    T: Order + Stored<Rt>,
    Rt: Runtime,
{
    let mut seen = elements(rt, s);
    let mut best = seen.next()?;
    for next in seen {
        if next.order(best) != std::cmp::Ordering::Less {
            best = next;
        }
    }
    Some(best.clone())
}

/// `part` matches `whole` from `offset` on.
fn matches_at<T, Rt>(
    rt: &Rt,
    whole: &[Erased<Rt, T>],
    part: &[Erased<Rt, T>],
    offset: usize,
) -> bool
where
    T: Order + Stored<Rt>,
    Rt: Runtime,
{
    (0..part.len()).all(|k| whole[offset + k].as_ref(rt) == part[k].as_ref(rt))
}

/// A total length past the address space is the overflow `Vec` itself
/// reports.
fn repeat_len(len: usize, times: usize) -> usize {
    let Some(total) = len.checked_mul(times) else {
        panic!("capacity overflow")
    };
    total
}

extern_signature! {
    ns: "slice",
    fn sort<T>(s: &mut [T])
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn contains<T>(s: &[T], x: &T) -> bool
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn binary_search<T>(s: &[T], x: &T) -> Option<u64>
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn is_sorted<T>(s: &[T]) -> bool
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn to_vec<T>(s: &[T]) -> Vec<T>
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn starts_with<T>(s: &[T], prefix: &[T]) -> bool
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn ends_with<T>(s: &[T], suffix: &[T]) -> bool
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn repeat<T>(s: &[T], times: u64) -> Vec<T>
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn min<T>(s: &[T]) -> Option<T>
    where
        T: Var<kind::Type>;
}

extern_signature! {
    ns: "slice",
    fn max<T>(s: &[T]) -> Option<T>
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
        min: $min:ident,
        max: $max:ident,
    ) => {
        #[extern_fn(instance_of = sort, effect = pure)]
        fn $sort<Rt>(ctx: &mut Ctx<'_, Rt>, s: &mut [Erased<Rt, $t>])
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            s.sort_by(|a, b| a.as_ref(rt).order(b.as_ref(rt)));
        }

        #[extern_fn(instance_of = contains, effect = pure)]
        fn $contains<Rt>(ctx: &mut Ctx<'_, Rt>, s: &[Erased<Rt, $t>], x: &$t) -> bool
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            elements(rt, s).any(|element| element == x)
        }

        #[extern_fn(instance_of = binary_search, effect = pure)]
        fn $binary_search<Rt>(ctx: &mut Ctx<'_, Rt>, s: &[Erased<Rt, $t>], x: &$t) -> Option<u64>
        where
            Rt: Runtime,
        {
            found(ctx.rt, s, x)
        }

        #[extern_fn(instance_of = is_sorted, effect = pure)]
        fn $is_sorted<Rt>(ctx: &mut Ctx<'_, Rt>, s: &[Erased<Rt, $t>]) -> bool
        where
            Rt: Runtime,
        {
            is_ordered(ctx.rt, s)
        }

        #[extern_fn(instance_of = to_vec, effect = pure)]
        fn $to_vec<Rt>(ctx: &mut Ctx<'_, Rt>, s: &[Erased<Rt, $t>]) -> Vec<$t>
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let mut out: Vec<$t> = Vec::with_capacity(s.len());
            for cell in s {
                out.push(cell.as_ref(rt).clone());
            }
            out
        }

        #[extern_fn(instance_of = starts_with, effect = pure)]
        fn $starts_with<Rt>(
            ctx: &mut Ctx<'_, Rt>,
            s: &[Erased<Rt, $t>],
            prefix: &[Erased<Rt, $t>],
        ) -> bool
        where
            Rt: Runtime,
        {
            prefix.len() <= s.len() && matches_at(ctx.rt, s, prefix, 0)
        }

        #[extern_fn(instance_of = ends_with, effect = pure)]
        fn $ends_with<Rt>(
            ctx: &mut Ctx<'_, Rt>,
            s: &[Erased<Rt, $t>],
            suffix: &[Erased<Rt, $t>],
        ) -> bool
        where
            Rt: Runtime,
        {
            let Some(offset) = s.len().checked_sub(suffix.len()) else {
                return false;
            };
            matches_at(ctx.rt, s, suffix, offset)
        }

        #[extern_fn(instance_of = repeat, effect = pure)]
        fn $repeat<Rt>(ctx: &mut Ctx<'_, Rt>, s: &[Erased<Rt, $t>], times: u64) -> Vec<$t>
        where
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let times = as_len(times);
            let mut out: Vec<$t> = Vec::with_capacity(repeat_len(s.len(), times));
            for _ in 0..times {
                for cell in s {
                    out.push(cell.as_ref(rt).clone());
                }
            }
            out
        }

        #[extern_fn(instance_of = min, effect = pure)]
        fn $min<Rt>(ctx: &mut Ctx<'_, Rt>, s: &[Erased<Rt, $t>]) -> Option<$t>
        where
            Rt: Runtime,
        {
            least(ctx.rt, s)
        }

        #[extern_fn(instance_of = max, effect = pure)]
        fn $max<Rt>(ctx: &mut Ctx<'_, Rt>, s: &[Erased<Rt, $t>]) -> Option<$t>
        where
            Rt: Runtime,
        {
            greatest(ctx.rt, s)
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
    min: min_int,
    max: max_int,
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
    min: min_index,
    max: max_index,
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
    min: min_float,
    max: max_float,
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
    min: min_bool,
    max: max_bool,
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
    min: min_str,
    max: max_str,
);

pub fn slice_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "slice",
        signatures: [
            sort, contains, binary_search, is_sorted, to_vec,
            starts_with, ends_with, repeat, min, max,
        ],
        fns: [
            len, is_empty, swap, get, first, last,
            rotate_left, rotate_right,
            sort_int, contains_int, binary_search_int, is_sorted_int,
            to_vec_int, starts_with_int, ends_with_int, repeat_int,
            min_int, max_int,
            sort_index, contains_index, binary_search_index, is_sorted_index,
            to_vec_index, starts_with_index, ends_with_index, repeat_index,
            min_index, max_index,
            sort_float, contains_float, binary_search_float, is_sorted_float,
            to_vec_float, starts_with_float, ends_with_float, repeat_float,
            min_float, max_float,
            sort_bool, contains_bool, binary_search_bool, is_sorted_bool,
            to_vec_bool, starts_with_bool, ends_with_bool, repeat_bool,
            min_bool, max_bool,
            sort_str, contains_str, binary_search_str, is_sorted_str,
            to_vec_str, starts_with_str, ends_with_str, repeat_str,
            min_str, max_str,
        ],
    }
}
