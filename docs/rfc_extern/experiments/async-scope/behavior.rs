//! The scope's runtime behaviour on the model: a call that ends, a scope
//! cancelled while its call is suspended, the same with a call that took a
//! lent value out across its waits, and a scope forgotten while its call is
//! suspended.

use model::{Exclusive, Program, Shared, Uniform, block_on, live, poll_once};

fn arguments() -> (Exclusive<i64>, Shared<String>) {
    (Exclusive(Uniform::new(1)), Shared(Uniform::new(String::from("abc"))))
}

/// Polls a scope over `program` once, while its call is suspended, and
/// drops it there.
fn cancel(program: Program) {
    let function = Uniform::new(program);
    let mut scope = std::pin::pin!(function.scope(arguments(), async |calling, (counter, text)| calling.call(counter, text).await));
    assert!(poll_once(scope.as_mut()).is_pending());
}

/// The scope gives the function and the arguments back with the result,
/// whether or not the call took a value out across its waits.
#[test]
fn a_scope_gives_its_arguments_back_with_the_result() {
    for takes_across_waits in [false, true] {
        let base = live();
        let function = Uniform::new(Program { yields: 2, takes_across_waits });
        let (function, (Exclusive(counter), Shared(text)), out) =
            block_on(function.scope(arguments(), async |calling, (counter, text)| calling.call(counter, text).await));
        assert_eq!(out, 3);
        assert_eq!(counter.get(), 2);
        assert_eq!(text.get(), "abc");
        drop((function, counter, text));
        assert_eq!(live(), base);
    }
}

/// Dropping the scope while its call is suspended abandons the call there:
/// nothing runs it apart, and its values are freed with the scope.
#[test]
fn a_cancelled_scope_abandons_its_call_and_frees_its_values() {
    let base = live();
    cancel(Program { yields: 3, takes_across_waits: false });
    assert_eq!(live(), base);
}

/// A call that took a lent value out across its waits, abandoned there,
/// leaves the place empty for its owner to release: the body the compiler
/// refuses (rfc_compiler ownership §4), and a double drop in an optimized
/// runtime.
#[test]
#[should_panic(expected = "a value released with its content taken out")]
fn a_cancelled_call_that_took_a_lent_value_out_leaves_its_place_empty() {
    cancel(Program { yields: 3, takes_across_waits: true });
}

/// Forgetting the scope while its call is suspended leaks what it holds,
/// which is safe; the call never resumes.
#[test]
fn a_forgotten_scope_leaks_and_its_call_never_resumes() {
    let base = live();
    let function = Uniform::new(Program { yields: 3, takes_across_waits: false });
    let mut scope = Box::pin(function.scope(arguments(), async |calling, (counter, text)| calling.call(counter, text).await));
    assert!(poll_once(scope.as_mut()).is_pending());
    std::mem::forget(scope);
    assert_eq!(live(), base + 3, "leaked, never freed");
}
