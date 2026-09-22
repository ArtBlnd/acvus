//! The standard registry's instances of the core signatures at the script
//! contract (RFC-0070 D5): one per word type, pinned to the meaning of the
//! instruction the compiler emits for that type; `Vec<T>`'s four, each over
//! the same signature at `T`; and the refusal where a type has none.

use acvus_interpreter::Value;
use acvus_interpreter_test::{Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn compile_and_run(source: &str, ret: Ty, opt: Opt) -> Result<Value, Refusal> {
    let i = Interner::new();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("main parses"));
    let cr = check_graph(
        &i,
        parsed,
        &[],
        &FxHashMap::default(),
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        ret,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        &i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()))
}

fn both(source: &str, ret: Ty) -> Value {
    let full = compile_and_run(source, ret.clone(), Opt::Full)
        .unwrap_or_else(|r| panic!("Opt::Full refused:\n  {}", r.messages.join("\n  ")));
    let none = compile_and_run(source, ret, Opt::None)
        .unwrap_or_else(|r| panic!("Opt::None refused:\n  {}", r.messages.join("\n  ")));
    assert_eq!(full.bits(), none.bits(), "the two optimization levels agree");
    full
}

fn int(source: &str) -> i64 {
    both(source, Ty::I64).as_int()
}

fn count(source: &str) -> i64 {
    both(source, Ty::U64).as_int()
}

fn boolean(source: &str) -> bool {
    both(source, Ty::Bool).as_bool()
}

fn refusal(source: &str, ret: Ty) -> String {
    match compile_and_run(source, ret, Opt::Full) {
        Ok(_) => panic!("the program was admitted"),
        Err(r) => r.messages.join(" | "),
    }
}

// -- The word types -----------------------------------------------------

#[test]
fn the_instances_at_int_answer_what_the_int_instructions_answer() {
    assert!(boolean("let a = 7; let b = 7; eq(&a, &b)"));
    assert!(!boolean("let a = 7; let b = 8; eq(&a, &b)"));
    assert_eq!(int("let a = 7; let b = 8; cmp(&a, &b)"), -1);
    assert_eq!(int("let a = 8; let b = 7; cmp(&a, &b)"), 1);
    assert_eq!(int("let a = 7; let b = 7; cmp(&a, &b)"), 0);
    assert_eq!(int("let a = 7; clone(&a)"), 7);
    assert!(boolean("let a = 7; let b = 7; hash(&a) == hash(&b)"));
}

/// `-0.0` is written as a product because the language has no unary minus.
const NEGATIVE_ZERO: &str = "let a = 0.0; let b = 0.0 * (0.0 - 1.0); ";

#[test]
fn eq_at_float_is_bit_equality_so_the_two_zeroes_differ() {
    assert!(!boolean(&format!("{NEGATIVE_ZERO} eq(&a, &b)")));
    assert!(boolean("let a = 1.5; let b = 1.5; eq(&a, &b)"));
}

#[test]
fn hash_at_float_separates_the_two_zeroes_as_eq_does() {
    assert!(!boolean(&format!("{NEGATIVE_ZERO} hash(&a) == hash(&b)")));
    assert!(boolean("let a = 1.5; let b = 1.5; hash(&a) == hash(&b)"));
}

#[test]
fn cmp_at_float_is_total_cmp_so_nan_orders() {
    assert_eq!(int("let a = f64::NAN(); let b = 1.0; cmp(&a, &b)"), 1);
    assert_eq!(int("let a = 1.0; let b = f64::NAN(); cmp(&a, &b)"), -1);
    assert_eq!(int(&format!("{NEGATIVE_ZERO} cmp(&b, &a)")), -1);
    assert_eq!(int("let a = 2.5; clone(&a) as i64"), 2);
}

#[test]
fn the_instances_at_bool_order_false_before_true() {
    assert_eq!(int("let a = false; let b = true; cmp(&a, &b)"), -1);
    assert_eq!(int("let a = true; let b = true; cmp(&a, &b)"), 0);
    assert!(boolean("let a = true; let b = true; eq(&a, &b)"));
    assert!(boolean("let a = true; clone(&a)"));
    assert!(boolean("let a = false; let b = false; hash(&a) == hash(&b)"));
}

#[test]
fn the_instances_at_string_order_bytewise_and_hash_equal_strings_alike() {
    assert_eq!(
        int(r#"let a = "abc".to_string(); let b = "abd".to_string(); cmp(&a, &b)"#),
        -1
    );
    assert_eq!(
        int(r#"let a = "ab".to_string(); let b = "a".to_string(); cmp(&a, &b)"#),
        1
    );
    assert!(boolean(
        r#"let a = "abc".to_string(); let b = "abc".to_string(); eq(&a, &b)"#
    ));
    assert!(
        boolean(r#"let a = "abc".to_string(); let b = "abc".to_string(); hash(&a) == hash(&b)"#),
        "two equal strings hash equal"
    );
    assert_eq!(
        count(r#"let a = "abc".to_string(); clone(&a).len()"#),
        3
    );
}

#[test]
fn the_instances_at_byte_answer_at_the_byte_type() {
    assert_eq!(int("let a = 1 as u8; let b = 2 as u8; cmp(&a, &b)"), -1);
    assert!(boolean("let a = 7 as u8; let b = 7 as u8; eq(&a, &b)"));
    assert!(boolean("let a = 7 as u8; clone(&a) as i64 == 7"));
    assert!(boolean(
        "let a = 7 as u8; let b = 7 as u8; hash(&a) == hash(&b)"
    ));
}

// -- `Vec<T>`, over the same signature at `T` ---------------------------

const NESTED_VECS: &str = "let outer = vec([vec([1, 2]), vec([3])]); ";

#[test]
fn clone_at_a_nested_vec_copies_through_the_element_s_own_clone() {
    assert!(
        boolean(&format!("{NESTED_VECS} let copy = clone(&outer); eq(&outer, &copy)")),
        "the copy holds what the original holds"
    );
    assert_eq!(
        int(&format!(
            "{NESTED_VECS} let copy = clone(&outer); copy[0][0] = 9; outer[0][0]"
        )),
        1,
        "a write into the copy's own element leaves the original"
    );
}

#[test]
fn eq_at_a_vec_of_strings_asks_the_string_instance_per_element() {
    assert!(boolean(
        r#"let a = vec(["x".to_string()]); let b = vec(["x".to_string()]); eq(&a, &b)"#
    ));
    assert!(!boolean(
        r#"let a = vec(["x".to_string()]); let b = vec(["y".to_string()]); eq(&a, &b)"#
    ));
    assert!(
        !boolean(r#"let a = vec(["x".to_string()]); let b = vec([]); eq(&a, &b)"#),
        "two lengths are two vectors"
    );
}

#[test]
fn cmp_at_a_vec_is_lexicographic_and_a_prefix_comes_first() {
    assert_eq!(int("let a = vec([1, 2]); let b = vec([1, 3]); cmp(&a, &b)"), -1);
    assert_eq!(int("let a = vec([1, 2]); let b = vec([1]); cmp(&a, &b)"), 1);
    assert_eq!(int("let a = vec([1, 2]); let b = vec([1, 2]); cmp(&a, &b)"), 0);
}

#[test]
fn hash_at_a_vec_is_the_elements_in_the_order_they_are_in() {
    assert!(boolean(
        "let a = vec([1, 2]); let b = vec([1, 2]); hash(&a) == hash(&b)"
    ));
    assert!(
        !boolean("let a = vec([1, 2]); let b = vec([2, 1]); hash(&a) == hash(&b)"),
        "the same elements in another order are another digest"
    );
}

// -- Where there is no instance -----------------------------------------

#[test]
fn a_clone_of_a_type_with_no_instance_is_refused_with_the_type_named() {
    let messages = refusal("let p = { x: 1, }; let q = clone(&p); q.x", Ty::I64);
    assert!(
        messages.contains("no instance of core::clone has the call type Fn(&{x: i64}) -> {x: i64}"),
        "the refusal names the signature and the type: {messages}"
    );
}
