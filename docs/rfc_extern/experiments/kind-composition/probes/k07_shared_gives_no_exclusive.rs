// expect: E0599
// A `Ref` gives `as_ref` alone (references §1).
use model::{Ref, Runtime};

pub fn upgrade<R>(text: Ref<'_, R, String>)
where
    R: Runtime,
{
    let _ = text.as_mut();
}
