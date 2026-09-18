use acvus_extern::{
    Arr, LenVar, Ref, RefMut, Registry, Runtime, Slice, SliceMut, TyVar, extern_fn, extern_registry,
};

#[extern_fn(effect = pure)]
fn len<T, N>(c: &Arr<T, N>) -> u64
where
    T: TyVar,
    N: LenVar,
{
    c.0.len() as u64
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
fn as_slice<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Rt>) -> Slice<T, Rt>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
fn as_slice_mut<T, N, Rt>(rt: &Rt, c: RefMut<Arr<T, N>, Rt>) -> SliceMut<T, Rt>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    SliceMut::of(c.elements_mut(rt))
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
        fns: [len, is_empty, as_slice, as_slice_mut, first, last],
    }
}
