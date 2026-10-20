// expect: pass
// A scope over a type variable's value, which carries no `'static` (variables
// §2), making a valid call: the loan is the trait's parameter, so the scope
// asks nothing of the arguments' own lifetimes.
use model::{ByMut, ByRef, Calling, Mut, Ref, Runtime, TypeVar, Uniform, scope};

fn call<'s, R, F, T>(_calling: Calling<'s, R, F>, _counter: Mut<'s, R, T>, _text: Ref<'s, R, T>)
where
    R: Runtime,
{
}

pub async fn run<R, F, T>(function: Uniform<R, F>, counter: T, text: T) -> (Uniform<R, F>, T, T)
where
    R: Runtime,
    T: TypeVar<R>,
{
    let (function, (ByMut(counter), ByRef(text)), ()) =
        scope(function, (ByMut(counter), ByRef(text)), async |calling, (counter, text)| call(calling, counter, text)).await;
    (function, counter, text)
}
