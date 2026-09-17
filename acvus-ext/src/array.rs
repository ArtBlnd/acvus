use acvus_extern::{
    Arr, LenVar, Ref, RefMut, Registry, Runtime, Trap, TyVar, extern_fn, extern_registry,
};

use crate::vec::checked_index;

#[extern_fn(effect = pure)]
fn len<T, N>(c: &Arr<T, N>) -> i64
where
    T: TyVar,
    N: LenVar,
{
    c.0.len() as i64
}

#[extern_fn(effect = pure)]
fn is_empty<T, N>(c: &Arr<T, N>) -> bool
where
    T: TyVar,
    N: LenVar,
{
    c.0.is_empty()
}

#[extern_fn(effect = pure)]
fn get<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Rt>, index: i64) -> Result<Ref<T, Rt>, Trap>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    let i = c.with(rt, |c| checked_index("get", c.0.len(), index))?;
    Ok(c.map(rt, |c| &c.0[i]))
}

#[extern_fn(effect = pure)]
fn get_mut<T, N, Rt>(rt: &Rt, c: RefMut<Arr<T, N>, Rt>, index: i64) -> Result<RefMut<T, Rt>, Trap>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    let i = c.with_mut(rt, |c| checked_index("get_mut", c.0.len(), index))?;
    Ok(c.map_mut(rt, |c| &mut c.0[i]))
}

#[extern_fn(effect = pure)]
fn first<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.0.first())
}

#[extern_fn(effect = pure)]
fn last<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.0.last())
}

// No `array::contains`: the reason is stated at `vec_registry`.

pub fn array_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "array",
        fns: [len, is_empty, get, get_mut, first, last],
    }
}
