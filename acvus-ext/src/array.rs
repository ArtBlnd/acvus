use acvus_extern::{
    Arr, Mut, Ref, Registry, Runtime, Shared, Slice, TransparentOver, Var, extern_fn,
    extern_registry, kind,
};

#[extern_fn(effect = pure)]
fn len<T, N>(c: &Arr<T, N>) -> u64
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
{
    c.0.len() as u64
}

#[extern_fn(effect = pure)]
fn is_empty<T, N>(c: &Arr<T, N>) -> bool
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
{
    c.0.is_empty()
}

#[extern_fn(effect = pure)]
fn as_slice<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Shared, Rt>) -> Slice<T, Shared, Rt>
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
fn as_slice_mut<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Mut, Rt>) -> Slice<T, Mut, Rt>
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
fn first<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Shared, Rt>) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    c.try_map(rt, |c| c.0.first())
}

#[extern_fn(effect = pure)]
fn last<T, N, Rt>(rt: &Rt, c: Ref<Arr<T, N>, Shared, Rt>) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
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
