//! The Deque instance of the context model: an append-only sequence that
//! holds the runtime's own values, exercised over struct-shaped elements.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[tokio::test]
async fn append_accumulates_over_structs() {
    let i = Interner::new();
    let src = "d = deque(); append(&mut d, { x: 1, }); append(&mut d, { x: 2, }); deque_len(d)";
    assert_eq!(run_script(&i, src, FxHashMap::default()).await, Value::Int(2));
}

#[tokio::test]
async fn get_reads_back_a_struct_field_in_order() {
    let i = Interner::new();
    let src = "d = deque(); append(&mut d, { x: 10, }); append(&mut d, { x: 20, }); deque_get(d, 1).x";
    assert_eq!(run_script(&i, src, FxHashMap::default()).await, Value::Int(20));
}

#[tokio::test]
async fn holds_enum_values() {
    let i = Interner::new();
    let src = "d = deque(); append(&mut d, Some(7)); append(&mut d, None); deque_len(d)";
    assert_eq!(run_script(&i, src, FxHashMap::default()).await, Value::Int(2));
}
