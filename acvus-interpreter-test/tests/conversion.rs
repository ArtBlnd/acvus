//! The conversion signatures (RFC-0019) at the contract: `to_string` and
//! `to_int` are one name each, read through a reference, with an instance
//! per scalar; a String that is not an integer is an error at the call.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

async fn text(src: &str) -> String {
    let i = Interner::new();
    run(&i, src, FxHashMap::default()).await
}

async fn int(src: &str) -> i64 {
    let i = Interner::new();
    run_script(&i, src, FxHashMap::default()).await.as_int()
}

fn byte_context(i: &Interner, value: u8) -> Context {
    FxHashMap::from_iter([(i.intern("b"), typed(Ty::Byte, Value::byte(value)))])
}

#[tokio::test]
async fn to_string_has_an_instance_for_every_scalar() {
    assert_eq!(text("{{ x = 42 }}{{ x.to_string() }}").await, "42");
    assert_eq!(text("{{ x = 1.5 }}{{ x.to_string() }}").await, "1.5");
    assert_eq!(text("{{ x = true }}{{ x.to_string() }}").await, "true");
    assert_eq!(text("{{ x = \"hi\" }}{{ x.to_string() }}").await, "hi");

    let i = Interner::new();
    assert_eq!(
        run(&i, "{{ @b.to_string() }}", byte_context(&i, 65)).await,
        "65"
    );
}

#[tokio::test]
async fn to_int_reads_every_scalar_through_a_reference() {
    assert_eq!(int("s = \"42\"; to_int(&s)").await, 42);
    assert_eq!(int("x = 1.9; x.to_int()").await, 1);
    assert_eq!(int("x = true; x.to_int()").await, 1);
    assert_eq!(int("x = 7; x.to_int()").await, 7);

    let i = Interner::new();
    assert_eq!(
        run_script(&i, "@b.to_int()", byte_context(&i, 65))
            .await
            .as_int(),
        65
    );
}

#[tokio::test]
#[should_panic(expected = "cannot parse string")]
async fn to_int_of_a_string_that_is_not_an_integer_is_an_error_at_the_call() {
    int("s = \"x\"; to_int(&s)").await;
}
