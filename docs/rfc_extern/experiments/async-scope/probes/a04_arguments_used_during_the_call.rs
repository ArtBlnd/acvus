// expect: E0382
// The caller uses an argument it gave the scope.
use model::*;
fn args() -> (Exclusive<i64>, Shared<String>) { (Exclusive(Uniform::new(1)), Shared(Uniform::new(String::from("abc")))) }

pub async fn run(function: Uniform<Program>) {
    let args = args();
    let scope = function.scope(args, async |calling, (counter, text)| calling.call(counter, text).await);
    let _ = args.0.0.get();
    scope.await;
}
