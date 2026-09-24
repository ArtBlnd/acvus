//! The functions on `Vec<T>`; the type itself is declared in
//! `acvus_extern`. An element read out of a borrowed container is a
//! reference into it, so the container is neither moved nor changed while
//! the element is in use (RFC-0028).

// SAFETY: each `unsafe(lent(..))` in this file asserts `NotKept` (RFC-0079
// rule 8) of a std container's or iterator stage's handler or type. Nothing
// here holds a static, a cell, a `#[state]` or a thread, and a value of a
// lent variable leaves a call only through an output its signature names.

use std::ops::Deref;

use acvus_extern::Ctx;
use acvus_extern::{
    Arr, Borrowable, Closure, ClosureFn, Instance, PassedByValue, Ref, Registry, Runtime, Shared,
    TransparentOver, Var, core, extern_fn, extern_registry, extern_signature, kind,
};

use crate::slice::{permute, swap_index};
use crate::word::verdict;

// A container demotes to a vec (RFC-0019).
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

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn reverse<T>(mut items: Vec<T>) -> Vec<T>
where
    T: Var<kind::Type>,
{
    items.reverse();
    items
}

#[extern_fn(instance_of = vec, effect = pure, unsafe(lent(T)))]
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
pub(crate) fn as_len(n: u64) -> usize {
    let Ok(n) = usize::try_from(n) else {
        panic!("capacity overflow")
    };
    n
}

/// The empty vec, `Vec::new`. It is `push`'s fold identity.
#[extern_fn(effect = pure, unsafe(lent(T)))]
fn new<T>() -> Vec<T>
where
    T: Var<kind::Type>,
{
    Vec::new()
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
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

#[extern_fn(effect = pure, ensures(ret = len(c)), unsafe(lent(T)))]
fn len<T>(c: &Vec<T>) -> u64
where
    T: Var<kind::Type>,
{
    c.len() as u64
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn is_empty<T>(c: &Vec<T>) -> bool
where
    T: Var<kind::Type>,
{
    c.is_empty()
}

/// The whole run of elements, borrowed in place (RFC-0047): the machine
/// indexes this and nothing else. No copy — the slice is a pointer and a
/// length into the container's own storage, returned as Rust's own borrow
/// of the parameter the caller lent (RFC-0068 rule 4).
#[extern_fn(effect = pure, unsafe(lent(T)))]
#[extern_view]
fn as_slice<T, Rt>(c: &Vec<T>) -> &[T]
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
#[extern_view]
fn as_slice_mut<T, Rt>(c: &mut Vec<T>) -> &mut [T]
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn first<T, Rt>(c: &Vec<T>) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c.first()
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn last<T, Rt>(c: &Vec<T>) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c.last()
}

/// Pushing a run of items onto `c` equals extending `c` by the vecs that
/// pushing each part of the run onto `new()` builds, in the run's order.
#[extern_fn(effect = pure, law(fold(combine = extend, identity = new)), unsafe(lent(T)))]
fn push<T>(c: &mut Vec<T>, item: T)
where
    T: Var<kind::Type>,
{
    c.push(item);
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn pop<T>(c: &mut Vec<T>) -> Option<T>
where
    T: Var<kind::Type>,
{
    c.pop()
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
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

#[extern_fn(effect = pure, unsafe(lent(T)))]
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

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn clear<T>(c: &mut Vec<T>)
where
    T: Var<kind::Type>,
{
    c.clear();
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn truncate<T>(c: &mut Vec<T>, len: u64)
where
    T: Var<kind::Type>,
{
    c.truncate(usize::try_from(len).unwrap_or(usize::MAX));
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn extend<T>(c: &mut Vec<T>, items: Vec<T>)
where
    T: Var<kind::Type>,
{
    c.extend(items);
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn swap<T>(c: &mut Vec<T>, i: u64, j: u64)
where
    T: Var<kind::Type>,
{
    let len = c.len();
    c.swap(swap_index(i, len), swap_index(j, len));
}

// -- the vec's own ------------------------------------------------------

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn capacity<T>(c: &Vec<T>) -> u64
where
    T: Var<kind::Type>,
{
    c.capacity() as u64
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn shrink_to_fit<T>(c: &mut Vec<T>)
where
    T: Var<kind::Type>,
{
    c.shrink_to_fit();
}

#[extern_fn(effect = pure, unsafe(lent(T)))]
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
// `Option<&mut T>`, and the language does not carry a `&mut` inside an
// `Option`: a store through what such a call binds is refused with "cannot
// store through &_: not a `&mut`". A script writes to an element through
// `v[i] = x`, which is the indexing instruction and keeps the exclusive
// loan at the place (RFC-0047).

#[extern_fn(effect = pure, unsafe(lent(T)))]
fn get<'a, T, Rt>(c: &Vec<T>, at: u64) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c.get(usize::try_from(at).ok()?)
}

// `sort_by` and `sort_by_key` stay the vec's own. Each is generic in `T`
// and reads no element at a type — it hands the closure a `&T` and moves
// the elements the verdicts order — but each is also named `vec::sort_by`
// by `docs/std/vec.md`, and a registry's namespace is the name a script
// writes.

type KeyOf<'a, T, E, Rt> = Closure<'a, (Ref<'a, T, Shared, Rt>,), i64, E, Rt>;

fn keyed_order(keys: Vec<i64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&at| keys[at]);
    order
}

fn sort_by_key_now<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, c: &mut Vec<T>, f: KeyOf<'_, T, E, Rt>)
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut keys: Vec<i64> = Vec::with_capacity(c.len());
    for at in 0..c.len() {
        keys.push(f.call_now(ctx, (&c[at],)));
    }
    permute(c, &keyed_order(keys));
}

#[extern_fn(effect = E, sync = sort_by_key_now, unsafe(lent(T)))]
async fn sort_by_key<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, c: &mut Vec<T>, f: KeyOf<'_, T, E, Rt>)
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut keys: Vec<i64> = Vec::with_capacity(c.len());
    for at in 0..c.len() {
        keys.push(f.call(ctx, (&c[at],)).await);
    }
    permute(c, &keyed_order(keys));
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
fn takes_left<'a>(verdict: i64) -> bool {
    verdict <= 0
}

type Comparator<'a, T, E, Rt> = Closure<'a, (Ref<'a, T, Shared, Rt>, Ref<'a, T, Shared, Rt>), i64, E, Rt>;

fn sort_by_now<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, c: &mut Vec<T>, f: Comparator<'_, T, E, Rt>)
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let n = c.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut merged: Vec<usize> = Vec::with_capacity(n);
    let mut width = 1usize;
    while width < n {
        for run in merge_runs(n, width) {
            let mut left = run.lo;
            let mut right = run.mid;
            merged.clear();
            while left < run.mid && right < run.hi {
                let verdict = f.call_now(ctx, (&c[order[left]], &c[order[right]]));
                let taken = if takes_left(verdict) {
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
    permute(c, &order);
}

#[extern_fn(effect = E, sync = sort_by_now, unsafe(lent(T)))]
async fn sort_by<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, c: &mut Vec<T>, f: Comparator<'_, T, E, Rt>)
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let n = c.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut merged: Vec<usize> = Vec::with_capacity(n);
    let mut width = 1usize;
    while width < n {
        for run in merge_runs(n, width) {
            let mut left = run.lo;
            let mut right = run.mid;
            merged.clear();
            while left < run.mid && right < run.hi {
                let verdict = f.call(ctx, (&c[order[left]], &c[order[right]])).await;
                let taken = if takes_left(verdict) {
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
    permute(c, &order);
}

// -- The core signatures at `Vec<T>`, over the same signature at `T` -----

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[extern_fn(instance_of = core::eq, effect = pure, unsafe(lent(T)))]
fn eq_vec<T, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: &Vec<T>,
    b: &Vec<T>,
    elem: Instance<'_, core::eq<T, Rt>, T, Rt>,
) -> bool
where
    T: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b) {
        if !elem.call(ctx, x, (y,)) {
            return false;
        }
    }
    true
}

#[extern_fn(instance_of = core::clone, effect = pure, unsafe(lent(T)))]
fn clone_vec<T, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: &Vec<T>,
    elem: Instance<'_, core::clone<T, Rt>, T, Rt>,
) -> Vec<T>
where
    T: Var<kind::Type>
        + Borrowable<Rt>
        + TransparentOver<Rt>
        + PassedByValue<Rt>
        + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut out = Vec::with_capacity(a.len());
    for x in a {
        out.push(elem.call(ctx, x, ()));
    }
    out
}

#[extern_fn(instance_of = core::cmp, effect = pure, unsafe(lent(T)))]
fn cmp_vec<T, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: &Vec<T>,
    b: &Vec<T>,
    elem: Instance<'_, core::cmp<T, Rt>, T, Rt>,
) -> i64
where
    T: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    for (x, y) in a.iter().zip(b) {
        let element = elem.call(ctx, x, (y,));
        if element != 0 {
            return element;
        }
    }
    verdict(a.len().cmp(&b.len()))
}

#[extern_fn(instance_of = core::hash, effect = pure, unsafe(lent(T)))]
fn hash_vec<T, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: &Vec<T>,
    elem: Instance<'_, core::hash<T, Rt>, T, Rt>,
) -> u64
where
    T: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut digest = FNV_OFFSET_BASIS;
    for x in a {
        digest = (digest ^ elem.call(ctx, x, ())).wrapping_mul(FNV_PRIME);
    }
    digest
}

pub fn vec_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "vec",
        types: [Vec<_>],
        signatures: [vec, filled],
        fns: [
            reverse, vec_array, new, with_capacity,
            filled_int, filled_float, filled_bool, filled_str,
            len, is_empty, as_slice, as_slice_mut, first, last,
            push, pop, insert, remove, clear, truncate, extend, swap,
            sort_by, sort_by_key, capacity, shrink_to_fit, split_off,
            get, eq_vec, clone_vec, cmp_vec, hash_vec,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed-seed linear congruential sequence (Knuth's MMIX constants),
    /// as in `num`'s law tests, so a failing law names the same inputs on
    /// every run.
    struct Samples(u64);

    impl Samples {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
    }

    fn pushed(onto: Vec<i64>, items: &[i64]) -> Vec<i64> {
        let mut state = onto;
        for &x in items {
            push(&mut state, x);
        }
        state
    }

    /// `push`'s fold law: over sampled runs `xs`, each split at every point
    /// into `a ++ b`, pushing `xs` onto `new()` equals `extend`-combining,
    /// in order, the states pushing `a` and pushing `b` onto `new()` reach.
    #[test]
    fn push_folds_by_extend_from_new() {
        let mut samples = Samples(0x5eed_1a55_0c1a_7e00);
        let edges = [i64::MIN, i64::MAX, 0, 1, -1];
        let mut runs: Vec<Vec<i64>> = vec![Vec::new(), edges.to_vec()];
        for len in 1..=24 {
            runs.push((0..len).map(|_| samples.next() as i64).collect());
        }
        for xs in &runs {
            let whole = pushed(new(), xs);
            assert_eq!(&whole, xs, "pushing onto `new()` builds the run itself");
            for at in 0..=xs.len() {
                let (a, b) = xs.split_at(at);
                let mut combined = pushed(new(), a);
                extend(&mut combined, pushed(new(), b));
                assert_eq!(combined, whole, "split of {xs:?} at {at}");
            }
        }
    }
}
