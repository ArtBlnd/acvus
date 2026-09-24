//! A store into a local slot is dead where no path from it reads the slot,
//! and what it stored goes with it.
//!
//! A slot `ssa_pass` promotes to a register carries no store into `dce` at
//! all, so each source below keeps its slot by lending it: the `+` on a
//! `String` takes `ref &s`, and a slot a reference names stays a slot.

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn listing(source: &str) -> String {
    let i = Interner::new();
    let context: FxHashMap<Astr, Ty> = [
        (i.intern("text"), Ty::String),
        (i.intern("other"), Ty::String),
        (i.intern("out"), Ty::String),
    ]
    .into_iter()
    .collect();
    compile_script_optimized(&i, source, &context).expect("the script compiles")
}

fn count(listing: &str, what: &str) -> usize {
    listing
        .lines()
        .filter(|line| line.contains(" | ") && line.contains(what))
        .count()
}

#[test]
fn a_store_the_next_read_names_stays() {
    let ir = listing("@out = @out + \")\"; 1");
    assert_eq!(
        count(&ir, "assign @out"),
        2,
        "the store the `ref &@out` reads and the store of the concatenation \
         both stand\n{ir}"
    );
}

/// A body that assigns `@out` whole before reading it does not fetch it
/// (RFC-0025 rule 2), so the store overwritten here is the one the call's
/// bracket fetches back after `peek` reads `@out`.
#[test]
fn a_store_overwritten_before_any_read_is_gone_and_its_value_released() {
    let ir = listing("let peek = | | -> @out + \"\"; peek(); @out = @other; @out = @out + \")\"; 1");
    let fetched = ir
        .lines()
        .skip_while(|line| *line != "=== main ===")
        .take_while(|line| !line.is_empty())
        .filter_map(|line| line.split_once(" = fetch @out"))
        .map(|(head, _)| head.split_whitespace().last().expect("a register name"))
        .last()
        .expect("the fetch after the call stands");
    assert_eq!(
        count(&ir, &format!("assign @out = {fetched}")),
        0,
        "the store of the value fetched after the call is dead\n{ir}"
    );
    assert_eq!(
        count(&ir, &format!("drop {fetched}")),
        1,
        "the fetched value the dead store would have held is released once\n{ir}"
    );
}

#[test]
fn a_dead_store_takes_its_pure_producer_with_it() {
    let ir = listing("let s = @text; s = @other; @out = s + \")\"; 1");
    assert_eq!(
        count(&ir, "assign s"),
        1,
        "the store of `@other`, which the `ref &s` reads, is the only one \
         left\n{ir}"
    );
    assert_eq!(
        count(&ir, "string_clone"),
        1,
        "the clone the dead store held had no other reader\n{ir}"
    );
}

#[test]
fn nothing_is_left_of_a_binding_no_one_reads() {
    let ir = listing("let s = \"ab\" + \"cd\"; 1");
    assert_eq!(count(&ir, "assign"), 0, "no store survives\n{ir}");
    assert_eq!(
        count(&ir, "string_concat"),
        0,
        "the concatenation the store held had no other reader\n{ir}"
    );
}
