// expect: lifetime may not live long enough
// A scope's loan is its own: the call's closure cannot give it out as its
// result (functions §3).
use model::{ByMut, ByRef, Runtime, Uniform, scope};

pub async fn run<R, F>(function: Uniform<R, F>, counter: Uniform<R, Uniform<R, i64>>, text: Uniform<R, Uniform<R, String>>)
where
    R: Runtime,
{
    let _ = scope(function, (ByMut(counter), ByRef(text)), async |_calling, (counter, _text)| counter).await;
}
