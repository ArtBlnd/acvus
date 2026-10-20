// expect: lifetime may not live long enough
// An exclusive scope's callable is its own: the call's closure cannot give
// it out as its result (functions §3).
use model::{ByRef, Runtime, Uniform, scope_mut};

pub async fn run<R, F>(function: Uniform<R, F>, text: Uniform<R, Uniform<R, String>>)
where
    R: Runtime,
{
    let _ = scope_mut(function, ByRef(text), async |calling, _text| calling).await;
}
