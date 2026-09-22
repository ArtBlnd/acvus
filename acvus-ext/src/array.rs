use acvus_extern::{
    Arr, Registry, Runtime, TransparentOver, Var, extern_fn, extern_registry, kind,
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
#[extern_view]
fn as_slice<T, N, Rt>(c: &Arr<T, N>) -> &[T]
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    &c.0
}

#[extern_fn(effect = pure)]
#[extern_view]
fn as_slice_mut<T, N, Rt>(c: &mut Arr<T, N>) -> &mut [T]
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    &mut c.0
}

#[extern_fn(effect = pure)]
fn first<T, N, Rt>(c: &Arr<T, N>) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    c.0.first()
}

#[extern_fn(effect = pure)]
fn last<T, N, Rt>(c: &Arr<T, N>) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    c.0.last()
}

#[extern_fn(effect = pure)]
fn get<T, N, Rt>(c: &Arr<T, N>, at: u64) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    c.0.get(usize::try_from(at).ok()?)
}

// The element surface is not declared here. `contains`, `binary_search`
// and `is_sorted` read an element at its own type, and every such operation
// lives once over `&[T]` in `crate::slice`; an array reaches it through the
// `as_slice` view below (RFC-0047 §5).

pub fn array_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "array",
        fns: [len, is_empty, as_slice, as_slice_mut, first, last, get],
    }
}
