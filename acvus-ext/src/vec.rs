//! The functions on `Vec<T>`; the type itself is declared in
//! `acvus_extern`. An element read out of a borrowed container is a
//! reference into it, so the container is neither moved nor changed while
//! the element is in use (RFC-0028).

use acvus_extern::{
    Arr, LenVar, Ref, RefMut, Registry, Runtime, Slice, SliceMut, TyVar, extern_fn,
    extern_registry, extern_signature,
};

// A container demotes to a vec (RFC-0027).
extern_signature! {
    ns: "std",
    fn vec<C, T>(items: C) -> Vec<T>
    where
        C: TyVar,
        T: TyVar;
}

pub(crate) fn checked_index(name: &'static str, len: usize, index: i64) -> usize {
    usize::try_from(index)
        .ok()
        .filter(|i| *i < len)
        .unwrap_or_else(|| panic!("{name}: index {index} is out of range for length {len}"))
}

#[extern_fn(effect = pure)]
fn reverse<T>(mut items: Vec<T>) -> Vec<T>
where
    T: TyVar,
{
    items.reverse();
    items
}

#[extern_fn(instance_of = vec, effect = pure)]
#[extern_cast]
fn vec_array<T, N>(items: Arr<T, N>) -> Vec<T>
where
    T: TyVar,
    N: LenVar,
{
    items.0
}

#[extern_fn(effect = pure)]
fn len<T>(c: &Vec<T>) -> i64
where
    T: TyVar,
{
    c.len() as i64
}

#[extern_fn(effect = pure)]
fn is_empty<T>(c: &Vec<T>) -> bool
where
    T: TyVar,
{
    c.is_empty()
}

#[extern_fn(effect = pure)]
fn get<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>, index: i64) -> Ref<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    let i = c.with(rt, |c| checked_index("get", c.len(), index));
    c.map(rt, |c| &c[i])
}

#[extern_fn(effect = pure)]
fn get_mut<T, Rt>(rt: &Rt, c: RefMut<Vec<T>, Rt>, index: i64) -> RefMut<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    let i = c.with_mut(rt, |c| checked_index("get_mut", c.len(), index));
    c.map_mut(rt, |c| &mut c[i])
}

/// The whole run of elements, borrowed in place (RFC-0047): the machine
/// indexes this and nothing else. No copy — the slice is a pointer and a
/// length into the container's own storage.
#[extern_fn(effect = pure)]
fn as_slice<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Slice<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
fn as_slice_mut<T, Rt>(rt: &Rt, c: RefMut<Vec<T>, Rt>) -> SliceMut<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    SliceMut::of(c.elements_mut(rt))
}

#[extern_fn(effect = pure)]
fn first<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.first())
}

#[extern_fn(effect = pure)]
fn last<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.last())
}

// There is no `vec::contains` (nor `array::` or `deque::contains`) beside
// `iter::contains`: a `Monomorphize` member's glue crosses every parameter
// naming the member through `CrossSpecialized`, and no form of a container
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
        signatures: [vec],
        fns: [
            reverse, vec_array, len, is_empty, get, get_mut, as_slice, as_slice_mut, first, last,
        ],
    }
}
