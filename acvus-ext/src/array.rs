use acvus_extern::Ctx;
use acvus_extern::{
    Arr, Mut, Ref, Registry, Runtime, Shared, Slice, TransparentOver, Var, extern_fn,
    extern_registry, extern_signature, kind,
};

use crate::vec::{elements, found};

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
fn as_slice<T, N, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Arr<T, N>, Shared, Rt>) -> Slice<T, Shared, Rt>
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
#[extern_view]
fn as_slice_mut<T, N, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Arr<T, N>, Mut, Rt>) -> Slice<T, Mut, Rt>
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    Slice::of(c.elements(rt))
}

#[extern_fn(effect = pure)]
fn first<T, N, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    c: Ref<Arr<T, N>, Shared, Rt>,
) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    c.try_map(rt, |c| c.0.first())
}

#[extern_fn(effect = pure)]
fn last<T, N, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    c: Ref<Arr<T, N>, Shared, Rt>,
) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    c.try_map(rt, |c| c.0.last())
}

#[extern_fn(effect = pure)]
fn get<T, N, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    c: Ref<Arr<T, N>, Shared, Rt>,
    at: u64,
) -> Option<Ref<T, Shared, Rt>>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let at = usize::try_from(at).ok()?;
    c.try_map(rt, |c| c.0.get(at))
}

// A fixed length admits the read-only half of the element surface, each a
// shared signature with one instance per concrete element type, for the
// reason stated at `vec_registry`.

extern_signature! {
    ns: "array",
    fn contains<T, N>(c: &Arr<T, N>, x: &T) -> bool
    where
        T: Var<kind::Type>,
        N: Var<kind::Length>;
}

extern_signature! {
    ns: "array",
    fn binary_search<T, N>(c: &Arr<T, N>, x: &T) -> Option<u64>
    where
        T: Var<kind::Type>,
        N: Var<kind::Length>;
}

macro_rules! read_of {
    (
        $t:ty,
        contains: $contains:ident,
        binary_search: $binary_search:ident,
    ) => {
        #[extern_fn(instance_of = contains, effect = pure)]
        fn $contains<N, Rt>(ctx: &mut Ctx<'_, Rt>, c: Ref<Arr<$t, N>, Shared, Rt>, x: &$t) -> bool
        where
            N: Var<kind::Length>,
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let run = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            // SAFETY: the run is this array's own values, every one erased
            // from `$t`, and the array is live for the call (RFC-0018).
            unsafe { elements::<$t, Rt>(rt, &run) }.any(|element| element == x)
        }

        #[extern_fn(instance_of = binary_search, effect = pure)]
        fn $binary_search<N, Rt>(
            ctx: &mut Ctx<'_, Rt>,
            c: Ref<Arr<$t, N>, Shared, Rt>,
            x: &$t,
        ) -> Option<u64>
        where
            N: Var<kind::Length>,
            Rt: Runtime,
        {
            let rt = ctx.rt;
            let run = Slice::<$t, Shared, Rt>::of(c.elements(rt)).into_elements();
            // SAFETY: as `$contains`.
            unsafe { found::<$t, Rt>(rt, &run, x) }
        }
    };
}

read_of!(
    i64,
    contains: contains_int,
    binary_search: binary_search_int,
);

read_of!(
    u64,
    contains: contains_index,
    binary_search: binary_search_index,
);

read_of!(
    f64,
    contains: contains_float,
    binary_search: binary_search_float,
);

read_of!(
    bool,
    contains: contains_bool,
    binary_search: binary_search_bool,
);

read_of!(
    String,
    contains: contains_str,
    binary_search: binary_search_str,
);

pub fn array_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "array",
        signatures: [contains, binary_search],
        fns: [
            len, is_empty, as_slice, as_slice_mut, first, last, get,
            contains_int, binary_search_int,
            contains_index, binary_search_index,
            contains_float, binary_search_float,
            contains_bool, binary_search_bool,
            contains_str, binary_search_str,
        ],
    }
}
