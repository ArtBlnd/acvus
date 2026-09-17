//! Reading a container (RFC-0028): the shared signatures `container::{len,
//! is_empty, get, get_mut, first, last}`, with instances for `Vec` and
//! `Array` here, for `Deque` in `deque`, and for `String` (`len` and
//! `is_empty` only) here.
//!
//! `container::get` has no `String` instance: `get` returns `&T` into the
//! container, and a `char` has no acvus type to be referenced as. A
//! character is read out by value with `string::char_at`.

use acvus_extern::{
    Arr, LenVar, Ref, RefMut, Registry, Runtime, Trap, TyVar, extern_fn, extern_registry,
};

pub mod sig {
    use acvus_extern::{Ref, RefMut, extern_signature};

    extern_signature! {
        ns: "container",
        fn len<C>(c: &C) -> i64
        where
            C: TyVar;
    }

    extern_signature! {
        ns: "container",
        fn is_empty<C>(c: &C) -> bool
        where
            C: TyVar;
    }

    extern_signature! {
        ns: "container",
        fn get<C, T, Rt>(c: &C, index: i64) -> Ref<T, Rt>
        where
            C: TyVar,
            T: TyVar,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "container",
        fn get_mut<C, T, Rt>(c: &mut C, index: i64) -> RefMut<T, Rt>
        where
            C: TyVar,
            T: TyVar,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "container",
        fn first<C, T, Rt>(c: &C) -> Option<Ref<T, Rt>>
        where
            C: TyVar,
            T: TyVar,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "container",
        fn last<C, T, Rt>(c: &C) -> Option<Ref<T, Rt>>
        where
            C: TyVar,
            T: TyVar,
            Rt: Runtime;
    }
}

pub(crate) fn checked_index(name: &'static str, len: usize, index: i64) -> Result<usize, Trap> {
    usize::try_from(index)
        .ok()
        .filter(|i| *i < len)
        .ok_or_else(|| Trap::call(name, format!("index {index} out of {len}")))
}

// -- Vec ----------------------------------------------------------------

#[extern_fn(instance_of = sig::len, effect = pure)]
fn len_vec<T>(c: &Vec<T>) -> i64
where
    T: TyVar,
{
    c.len() as i64
}

#[extern_fn(instance_of = sig::is_empty, effect = pure)]
fn is_empty_vec<T>(c: &Vec<T>) -> bool
where
    T: TyVar,
{
    c.is_empty()
}

#[extern_fn(instance_of = sig::get, effect = pure)]
fn get_vec<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>, index: i64) -> Result<Ref<T, Rt>, Trap>
where
    T: TyVar,
    Rt: Runtime,
{
    let i = c.with(rt, |c| checked_index("get", c.len(), index))?;
    Ok(c.map(rt, |c| &c[i]))
}

#[extern_fn(instance_of = sig::get_mut, effect = pure)]
fn get_mut_vec<T, Rt>(rt: &Rt, c: RefMut<Vec<T>, Rt>, index: i64) -> Result<RefMut<T, Rt>, Trap>
where
    T: TyVar,
    Rt: Runtime,
{
    let i = c.with_mut(rt, |c| checked_index("get_mut", c.len(), index))?;
    Ok(c.map_mut(rt, |c| &mut c[i]))
}

#[extern_fn(instance_of = sig::first, effect = pure)]
fn first_vec<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.first())
}

#[extern_fn(instance_of = sig::last, effect = pure)]
fn last_vec<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.last())
}

// -- Array --------------------------------------------------------------

#[extern_fn(instance_of = sig::len, effect = pure)]
fn len_array<T, N>(c: &Arr<T, N>) -> i64
where
    T: TyVar,
    N: LenVar,
{
    c.0.len() as i64
}

#[extern_fn(instance_of = sig::is_empty, effect = pure)]
fn is_empty_array<T, N>(c: &Arr<T, N>) -> bool
where
    T: TyVar,
    N: LenVar,
{
    c.0.is_empty()
}

#[extern_fn(instance_of = sig::get, effect = pure)]
fn get_array<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Rt>, index: i64) -> Result<Ref<T, Rt>, Trap>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    let i = c.with(rt, |c| checked_index("get", c.0.len(), index))?;
    Ok(c.map(rt, |c| &c.0[i]))
}

#[extern_fn(instance_of = sig::get_mut, effect = pure)]
fn get_mut_array<T, N, Rt>(
    rt: &Rt,
    c: RefMut<Arr<T, N>, Rt>,
    index: i64,
) -> Result<RefMut<T, Rt>, Trap>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    let i = c.with_mut(rt, |c| checked_index("get_mut", c.0.len(), index))?;
    Ok(c.map_mut(rt, |c| &mut c.0[i]))
}

#[extern_fn(instance_of = sig::first, effect = pure)]
fn first_array<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.0.first())
}

#[extern_fn(instance_of = sig::last, effect = pure)]
fn last_array<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.0.last())
}

// -- String -------------------------------------------------------------

/// The length in characters (`chars().count()`); `string::len_str` is the
/// length in bytes.
#[extern_fn(instance_of = sig::len, effect = pure)]
fn len_string(c: &String) -> i64 {
    c.chars().count() as i64
}

#[extern_fn(instance_of = sig::is_empty, effect = pure)]
fn is_empty_string(c: &String) -> bool {
    c.is_empty()
}

pub fn container_registry<Rt>() -> Registry<Rt>
where
    Rt: Runtime,
{
    extern_registry! {
        ns: "std",
        signatures: [sig::len, sig::is_empty, sig::get, sig::get_mut, sig::first, sig::last],
        fns: [
            len_vec, is_empty_vec, get_vec, get_mut_vec, first_vec, last_vec,
            len_array, is_empty_array, get_array, get_mut_array, first_array, last_array,
            len_string, is_empty_string,
        ],
    }
}
