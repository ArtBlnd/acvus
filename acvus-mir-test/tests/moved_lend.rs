//! A storage is read only while the part read is alive (RFC-0018): a `Take`
//! of a place in it, a reference to one. A reference to a moved place is a
//! use after move, refused where the source wrote it.
//!
//! The harness is `compile_script_mode_optimized`: lowering, then the move
//! and borrow checks, then the optimization pipeline and every validation.
//! The move check runs in both of those, so a program the rule refuses is
//! refused once per pass and the error text arrives more than once.

use acvus_mir_test::compile_script_mode_optimized;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn errors(source: &str) -> Vec<String> {
    let i = Interner::new();
    match compile_script_mode_optimized(&i, source, &FxHashMap::default()) {
        Ok(ir) => panic!("expected a validation error, compiled:\n{ir}"),
        Err(e) => e.lines().map(str::to_string).collect(),
    }
}

fn use_after_move_of_a(source: &str) {
    let errs = errors(source);
    assert!(!errs.is_empty(), "{source}");
    assert!(
        errs.iter()
            .all(|e| e.contains("use of `a` after it was moved")),
        "{source}: {errs:#?}"
    );
}

#[test]
fn a_take_of_a_moved_storage_is_refused() {
    use_after_move_of_a("let a = [1, 2]; let b = a; a");
}

#[test]
fn a_method_receiver_lent_from_a_moved_storage_is_refused() {
    use_after_move_of_a("let a = [1, 2]; let b = a; a.len()");
}

#[test]
fn a_reference_to_a_moved_storage_is_refused() {
    use_after_move_of_a("let a = [1, 2]; let b = a; len(&a)");
}

#[test]
fn a_reference_bound_from_a_moved_storage_is_refused() {
    use_after_move_of_a("let a = [1, 2]; let b = a; let r = &a; *get(r, 0)");
}

#[test]
fn a_reference_to_a_storage_assigned_back_is_accepted() {
    let i = Interner::new();
    compile_script_mode_optimized(
        &i,
        "let a = [1, 2]; let b = a; a = [3, 4]; a.len()",
        &FxHashMap::default(),
    )
    .expect("an assignment revives the storage");
}
