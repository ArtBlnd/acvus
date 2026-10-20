// expect: E0382
// An owning scope consumes its function: it is not there to use after.
use model::{ByRef, Runtime, Uniform, scope_once};

pub async fn run<R, F>(function: Uniform<R, F>, text: Uniform<R, Uniform<R, String>>)
where
    R: Runtime,
{
    let _ = scope_once(function, ByRef(text), async |_calling, _text| ()).await;
    drop(function);
}
