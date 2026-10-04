// expect: lifetime may not live long enough
// An inner scope's call is given a loan of the outer scope.
use model::*;
fn args() -> (Exclusive<i64>, Shared<String>) { (Exclusive(Uniform::new(1)), Shared(Uniform::new(String::from("abc")))) }

pub async fn run(outer: Uniform<Program>, inner: Uniform<Program>) {
    let _ = outer.scope(args(), async move |_calling, (counter, _text)| {
        inner.scope(Shared(Uniform::new(String::from("x"))), async move |calling, text| calling.call(counter, text).await).await
    }).await;
}
