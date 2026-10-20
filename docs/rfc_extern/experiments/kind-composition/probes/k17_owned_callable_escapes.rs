// expect: lifetime may not live long enough
// An owning scope's callable is its own too: given to the call, it cannot
// leave the scope as its result, so the function is consumed or released
// inside it (functions §3).
use model::{ByRef, Runtime, Uniform, scope_once};

pub async fn run<R, F>(function: Uniform<R, F>, text: Uniform<R, Uniform<R, String>>)
where
    R: Runtime,
{
    let _ = scope_once(function, ByRef(text), async |calling, _text| calling).await;
}
