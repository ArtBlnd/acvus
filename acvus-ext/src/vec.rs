//! The functions on `Vec<T>`; the type itself is declared in
//! `acvus_extern`. An element read out of a borrowed container is a
//! reference into it, so the container is neither moved nor changed while
//! the element is in use (RFC-0028).

use acvus_extern::{
    Arr, Ref, RefMut, Registry, Runtime, Slice, SliceMut, TransparentOver, Var, extern_fn,
    extern_registry, extern_signature, kind,
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
fn as_slice<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Slice<T, Rt>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
fn as_slice_mut<T, Rt>(rt: &Rt, c: RefMut<Vec<T>, Rt>) -> SliceMut<T, Rt>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    SliceMut::of(c.elements_mut(rt))
}

#[extern_fn(effect = pure)]
fn first<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Option<Ref<T, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.first())
}

#[extern_fn(effect = pure)]
fn last<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Option<Ref<T, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
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

// There is no `vec::contains` (nor `array::` or `deque::contains`) beside
// `iter::contains`: a `Monomorphize` member's glue crosses every parameter
// naming the member at its specialized representation, and no form of a container
// of `Erased<Rt, T>` has one that reads the storage — `Vec` and `Deque`
// cross whole, so the payload's `TypeId` is the container of values, not
// of `Erased`; `Arr` crosses per element, which `Erased` is not; `Ref<C,
// Rt>` has no implementation. `Iter` has, as an extension type stored as
// its payload. Until `acvus-extern` gives a container of an erased element
// a crossing, a container is searched as `into_iter(xs) | contains(x)`,
// which the checker settles through the declared cast (RFC-0043).

pub fn vec_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "vec",
        types: [Vec<_>],
        signatures: [vec, filled],
        fns: [
            reverse, vec_array, with_capacity,
            filled_int, filled_float, filled_bool, filled_str,
            len, is_empty, as_slice, as_slice_mut, first, last,
            push, pop, insert, remove, clear, truncate, extend, swap,
        ],
    }
}
