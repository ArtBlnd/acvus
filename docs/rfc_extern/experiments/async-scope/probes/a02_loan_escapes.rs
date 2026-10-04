// expect: lifetime may not live long enough
// The closure gives its loan out as its result.
use model::*;
fn args() -> (Exclusive<i64>, Shared<String>) { (Exclusive(Uniform::new(1)), Shared(Uniform::new(String::from("abc")))) }

pub async fn run(function: Uniform<Program>) {
    let _ = function.scope(args(), async |_calling, (counter, _text)| counter).await;
}
