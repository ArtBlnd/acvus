//! Intent tests for the liveness of a storage lent into a binding: a
//! reference names what the storage it reaches names, so a local vector an
//! iterator borrows is dropped after the loop that uses the iterator, never
//! before it (RFC-0018).

use acvus_mir_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn main_body(ir: &str) -> String {
    ir.split("=== main ===")
        .nth(1)
        .expect("main body")
        .split("=== ")
        .next()
        .expect("main body")
        .to_string()
}

/// This reads the `; rN (name) : Ty` lines of the IR printer in
/// `acvus_mir`. That format is not this crate's to choose: change the
/// printer and this function stops finding anything, while everything here
/// still compiles.
fn register_of(ir: &str, name: &str) -> String {
    let tag = format!(" ({name}) : ");
    let line = ir
        .lines()
        .find(|l| l.trim_start().starts_with("; r") && l.contains(&tag))
        .unwrap_or_else(|| panic!("no register bound to `{name}` in:\n{ir}"));
    line.trim_start()
        .trim_start_matches("; ")
        .split(' ')
        .next()
        .expect("register")
        .to_string()
}

fn position_of(body: &str, needle: &str) -> usize {
    body.find(needle)
        .unwrap_or_else(|| panic!("`{needle}` is absent from:\n{body}"))
}

const WHILE_LET_OVER_A_LOCAL_VEC: &str = "let v = range(0, 10) | collect; \
     let it = as_iter(&v); \
     let acc = 0; \
     while let Some(x) = next(&mut it) { acc = acc + *x; } \
     acc";

#[test]
fn a_vector_an_iterator_borrows_is_dropped_after_the_loop() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(&i, WHILE_LET_OVER_A_LOCAL_VEC, &FxHashMap::default())
        .expect("the program compiles");
    let body = main_body(&ir);
    let drop_v = format!("drop {}", register_of(&ir, "v"));

    let loop_exit = position_of(&body, "jump L0(");
    let dropped = position_of(&body, &drop_v);
    assert!(
        dropped > loop_exit,
        "`{drop_v}` must follow the loop that reads through the iterator, not precede it:\n{body}"
    );
}

#[test]
fn a_vector_two_iterators_borrow_outlives_both_loops() {
    let i = Interner::new();
    let source = "let v = range(0, 10) | collect; \
         let a = as_iter(&v); \
         let b = as_iter(&v); \
         let acc = 0; \
         while let Some(x) = next(&mut a) { acc = acc + *x; } \
         while let Some(y) = next(&mut b) { acc = acc + *y; } \
         acc";
    let ir = compile_script_mode_optimized(&i, source, &FxHashMap::default())
        .expect("the program compiles");
    let body = main_body(&ir);
    let drop_v = format!("drop {}", register_of(&ir, "v"));

    let second_loop = body.rfind("jump L").expect("a second loop");
    assert!(
        position_of(&body, &drop_v) > second_loop,
        "`{drop_v}` must follow both loops:\n{body}"
    );
}
