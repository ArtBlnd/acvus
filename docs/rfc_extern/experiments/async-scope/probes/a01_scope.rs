// expect: pass
// A scope lends the function and the arguments to a call, and gives them back.
use model::*;
fn args() -> (Exclusive<i64>, Shared<String>) { (Exclusive(Uniform::new(1)), Shared(Uniform::new(String::from("abc")))) }

pub async fn run(function: Uniform<Program>) -> usize {
    let (_function, (Exclusive(counter), _text), out) = function.scope(args(), async |calling, (counter, text)| calling.call(counter, text).await).await;
    out + counter.get() as usize
}
