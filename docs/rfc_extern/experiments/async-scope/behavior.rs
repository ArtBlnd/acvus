//! The scope's runtime behaviour on the model: a call that ends, a scope
//! cancelled while its call runs, and a scope forgotten while its call runs.

use model::{Exclusive, Program, Shared, Uniform, block_on, live, poll_once, run_tasks};

fn arguments() -> (Exclusive<i64>, Shared<String>) {
    (Exclusive(Uniform::new(1)), Shared(Uniform::new(String::from("abc"))))
}

/// The scope gives the function and the arguments back with the result.
#[test]
fn a_scope_gives_its_arguments_back_with_the_result() {
    let base = live();
    let function = Uniform::new(Program { yields: 2 });
    let (function, (Exclusive(counter), Shared(text)), out) =
        block_on(function.scope(arguments(), async |calling, (counter, text)| calling.call(counter, text).await));
    assert_eq!(out, 3);
    assert_eq!(counter.get(), 2);
    assert_eq!(text.get(), "abc");
    drop((function, counter, text));
    assert_eq!(live(), base);
}

/// Dropping the scope while its call runs ends its owners, and the runtime
/// keeps what the call uses until the call ends, then frees it.
#[test]
fn a_cancelled_scope_leaves_its_values_to_the_runtime_until_the_call_ends() {
    let base = live();
    let function = Uniform::new(Program { yields: 3 });
    {
        let mut scope = std::pin::pin!(function.scope(arguments(), async |calling, (counter, text)| calling.call(counter, text).await));
        assert!(poll_once(scope.as_mut()).is_pending());
    }
    assert_eq!(live(), base + 3, "the running call's values are kept");
    while run_tasks() {}
    assert_eq!(live(), base, "and freed once it ends");
}

/// Forgetting the scope while its call runs leaks what it holds, which is
/// safe; the call runs to its end on values still alive.
#[test]
fn a_forgotten_scope_leaks_and_its_call_ends_on_live_values() {
    let base = live();
    let function = Uniform::new(Program { yields: 3 });
    let mut scope = Box::pin(function.scope(arguments(), async |calling, (counter, text)| calling.call(counter, text).await));
    assert!(poll_once(scope.as_mut()).is_pending());
    std::mem::forget(scope);
    while run_tasks() {}
    assert_eq!(live(), base + 3, "leaked, never freed");
}
