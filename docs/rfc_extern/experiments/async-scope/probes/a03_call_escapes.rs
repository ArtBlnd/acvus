// expect: lifetime may not live long enough
// The closure gives the started call out, unawaited, as its result.
use model::*;
fn args() -> (Exclusive<i64>, Shared<String>) { (Exclusive(Uniform::new(1)), Shared(Uniform::new(String::from("abc")))) }

pub async fn run(function: Uniform<Program>) {
    let _ = function.scope(args(), async |calling, (counter, text)| calling.call(counter, text)).await;
}
