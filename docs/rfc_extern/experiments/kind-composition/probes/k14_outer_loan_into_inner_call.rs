// expect: lifetime may not live long enough
// A loan from outside a scope fits none of its calls: an inner scope's call
// is given the outer scope's loan (functions §3).
use model::{ByMut, ByRef, Calling, Mut, Runtime, Uniform, scope};

/// A call that takes an exclusive loan of its own scope.
fn call<'s, R, F, T>(_calling: Calling<'s, R, F>, _counter: Mut<'s, R, T>)
where
    R: Runtime,
{
}

pub async fn run<R, F, T>(outer: Uniform<R, F>, inner: Uniform<R, F>, counter: T, text: T, other: T)
where
    R: Runtime,
    T: model::TypeVar<R>,
{
    let _ = scope(outer, (ByMut(counter), ByRef(text)), async move |_calling, (counter, _text)| {
        scope(inner, ByRef(other), async move |calling, _other| call(calling, counter)).await;
    })
    .await;
}
