// expect: pass
// A scope over arguments whose type is a variable of the caller.
use model::*;
pub async fn run<A: Lend, O>(function: Uniform<Program>, args: A, start: impl for<'s> AsyncFnOnce(Calling<'s>, A::Lent<'s>) -> O) -> (A, O) {
    let (_function, args, out) = function.scope(args, start).await;
    (args, out)
}
