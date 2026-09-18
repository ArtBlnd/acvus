//! RFC-0047 amended, rule 1: a slice-typed value is two adjacent word
//! registers, and `Index` reads the second of them as the length.
//!
//! `Body::slot_kinds` is what `machine::open_frame` writes, so it is where
//! the pair's two registers show as one word class.

use acvus_interpreter::code::{Body, Op};
use acvus_interpreter::index_handlers::Read;
use acvus_interpreter_test::listing::{main_body, prepared_script};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// An index is a `u64` and the language has no `u64` literal (RFC-0047,
/// "what it costs"), so the index is derived from `len`.
const READ_IN_RANGE: &str = "\
let a = vec([10, 20, 30, 40]); \
let n = len(&a); \
let i = n - n; \
a[i]";

const READ_PAST_THE_END: &str = "\
let a = vec([10, 20]); \
let n = len(&a); \
a[n]";

fn reads_of(body: &Body) -> Vec<Read> {
    let mut found = Vec::new();
    for head in &body.heads {
        walk(head.as_ref(), &mut found);
    }
    found
}

fn walk(op: &dyn Op, found: &mut Vec<Read>) {
    if let Some(read) = op.index_read() {
        found.push(read);
    }
    for owned in op.owns() {
        walk(owned.head, found);
    }
    if let Some(next) = op.successor() {
        walk(next, found);
    }
}

#[tokio::test]
async fn a_slice_is_two_adjacent_registers_of_one_word_class() {
    let interner = Interner::new();
    let prepared = prepared_script(&interner, READ_IN_RANGE, Context::default(), Ty::I64);
    let body = main_body(&prepared);

    let reads = reads_of(body);
    let read = match reads.as_slice() {
        [read] => *read,
        other => panic!(
            "the script prepares one bound-checked read, not {}",
            other.len()
        ),
    };

    let ptr = read.slice.ptr;
    let len = read.slice.len;
    assert_eq!(
        len.index(),
        ptr.index() + 1,
        "the length register is the one after the pointer's"
    );

    let opened: Vec<usize> = body
        .slot_kinds
        .iter()
        .filter(|k| k.slot == ptr || k.slot == len)
        .map(|k| k.slot.index())
        .collect();
    assert_eq!(
        opened,
        [ptr.index(), len.index()],
        "both registers of the pair are opened by the frame, in a frame of {} \
         whose word class is {:?}",
        body.frame_len,
        body.slot_kinds
            .iter()
            .map(|k| (k.slot.index(), k.kind))
            .collect::<Vec<_>>()
    );
    assert!(
        body.slot_kinds
            .iter()
            .filter(|k| k.slot == ptr || k.slot == len)
            .all(|k| k.kind.is_inline()),
        "a pair register is opened at an inline kind, as every word register is"
    );
    assert_ne!(
        read.index, ptr,
        "the index rides in a register of its own, not in the pair"
    );
    assert_ne!(read.index, len, "nor in the pair's second register");
}

#[tokio::test]
async fn a_read_in_range_reaches_the_element() {
    let interner = Interner::new();
    let value = run_script_mode(&interner, READ_IN_RANGE, Context::default(), Ty::I64).await;
    assert_eq!(
        value.as_int(),
        10,
        "the pointer register named the container's first element"
    );
}

#[tokio::test]
#[should_panic(expected = "index out of bounds: the len is 2 but the index is 2")]
async fn the_bound_check_reads_the_length_from_the_second_register() {
    let interner = Interner::new();
    run_script_mode(&interner, READ_PAST_THE_END, Context::default(), Ty::I64).await;
}
