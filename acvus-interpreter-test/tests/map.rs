//! `HashMap<K, V>` and `HashSet<K>` at the script contract.
//!
//! `hash_map()` and `hash_set()` take the key type's own `core::hash` and
//! `core::eq`; `hash_map_by` and `hash_set_by` take the two as lambdas,
//! which is where a key with no instances goes. In a lambda the comparison
//! is written `a == b` over the two references rather than `*a == *b`:
//! dereferencing moves a `String` out of the reference the comparator was
//! lent, and the checker refuses that.

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

/// The value a program yields at both optimization levels. Disagreement is
/// the differential's own contract, so it fails here.
fn both(source: &str, ret: Ty) -> Value {
    let full = compile_and_run(source, ret.clone(), Opt::Full)
        .unwrap_or_else(|r| panic!("Opt::Full refused:\n  {}", r.messages.join("\n  ")));
    let none = compile_and_run(source, ret, Opt::None)
        .unwrap_or_else(|r| panic!("Opt::None refused:\n  {}", r.messages.join("\n  ")));
    assert_eq!(
        full.bits(),
        none.bits(),
        "the two optimization levels agree"
    );
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

fn refusal_at_both(source: &str, ret: Ty) -> String {
    let full = refusal(source, ret.clone());
    let none = match compile_and_run(source, ret, Opt::None) {
        Ok(_) => panic!("the program was admitted at `Opt::None`"),
        Err(r) => r.messages.join(" | "),
    };
    assert_eq!(full, none, "the two optimization levels refuse alike");
    full
}

const KEYING: &str = "|k| -> hash(k), |a, b| -> a == b";

fn with_map(body: &str) -> String {
    format!("let m = hash_map_by({KEYING}); {body}")
}

fn with_set(body: &str) -> String {
    format!("let s = hash_set_by({KEYING}); {body}")
}

// -- The constructors ---------------------------------------------------

#[test]
fn a_map_the_key_type_of_which_is_known_and_that_holds_nothing_is_empty() {
    assert!(boolean(&with_map(
        "insert(&mut m, 1, 10); let q = 1; remove(&mut m, &q); is_empty(&m)"
    )));
}

#[test]
fn with_capacity_holds_the_same_map_a_bare_constructor_does() {
    assert_eq!(
        count(&format!(
            "let m = with_capacity(8u64, {KEYING}); insert(&mut m, 1, 10); len(&m)"
        )),
        1
    );
}

/// The constructors that require rather than take: `K` is settled by the
/// first `insert`, and the requirement is decided there.
#[test]
fn a_map_over_the_key_s_own_instances_finds_what_it_inserted() {
    assert_eq!(
        int(
            "let m = hash_map(); insert(&mut m, 1, 10); insert(&mut m, 2, 20); \
             let q = 2; *get(&m, &q).unwrap()"
        ),
        20
    );
    assert_eq!(
        count("let m = hash_map(); insert(&mut m, 1, 10); insert(&mut m, 1, 20); len(&m)"),
        1
    );
}

#[test]
fn a_string_keyed_map_over_the_key_s_own_instances_finds_an_equal_string() {
    assert_eq!(
        int(
            "let m = hash_map(); let w = \"alpha\".to_string(); insert(&mut m, w, 7); \
             let q = \"alpha\".to_string(); *get(&m, &q).unwrap()"
        ),
        7
    );
    assert_eq!(
        int(
            "let m = hash_map(); let w = \"alpha\".to_string(); insert(&mut m, w, 7); \
             let q = \"beta\".to_string(); \
             let found = if let Some(v) = get(&m, &q) { *v } else { 0 - 1 }; found"
        ),
        -1
    );
}

#[test]
fn a_set_over_the_key_s_own_instances_keeps_one_of_each_key() {
    assert_eq!(
        count("let s = hash_set(); insert(&mut s, 1); insert(&mut s, 1); insert(&mut s, 2); len(&s)"),
        2
    );
    assert!(boolean(
        "let s = hash_set(); insert(&mut s, 1); let q = 1; contains(&s, &q)"
    ));
}

/// An object has no instance of either signature, so its keying is the pair
/// of lambdas.
#[test]
fn an_object_key_is_kept_by_the_lambdas_hash_map_by_takes() {
    assert_eq!(
        int(
            "let m = hash_map_by(|k| -> k.x as u64, |a, b| -> a.x == b.x); \
             insert(&mut m, { x: 1, }, 10); insert(&mut m, { x: 2, }, 20); \
             let q = { x: 2, }; *get(&m, &q).unwrap()"
        ),
        20
    );
}

// -- insert, get, contains_key, len -------------------------------------

#[test]
fn insert_of_a_fresh_key_answers_none_and_grows_the_map() {
    assert_eq!(
        count(&with_map("let old = insert(&mut m, 1, 10); len(&m)")),
        1
    );
    assert_eq!(
        int(&with_map("insert(&mut m, 1, 10).unwrap_or(0 - 1)")),
        -1,
        "a fresh key replaced no value"
    );
}

#[test]
fn insert_over_a_key_answers_the_value_it_replaced() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 1, 10); insert(&mut m, 1, 20).unwrap()"
        )),
        10
    );
    assert_eq!(
        count(&with_map(
            "insert(&mut m, 1, 10); insert(&mut m, 1, 20); len(&m)"
        )),
        1,
        "a replaced key is one entry"
    );
}

#[test]
fn get_reads_the_value_a_key_names() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 1, 10); insert(&mut m, 2, 20); let q = 2; *get(&m, &q).unwrap()"
        )),
        20
    );
}

#[test]
fn get_of_an_absent_key_is_none() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 1, 10); let q = 9; \
             let found = if let Some(v) = get(&m, &q) { *v } else { 0 - 1 }; found"
        )),
        -1
    );
}

#[test]
fn a_string_key_is_found_by_an_equal_string() {
    assert_eq!(
        int(&with_map(
            "let w = \"alpha\".to_string(); insert(&mut m, w, 7); \
             let q = \"alpha\".to_string(); *get(&m, &q).unwrap()"
        )),
        7
    );
}

#[test]
fn contains_key_answers_for_a_key_present_and_one_absent() {
    assert!(boolean(&with_map(
        "insert(&mut m, 1, 10); let q = 1; contains_key(&m, &q)"
    )));
    assert!(!boolean(&with_map(
        "insert(&mut m, 1, 10); let q = 2; contains_key(&m, &q)"
    )));
}

// -- get_mut, or_insert -------------------------------------------------

/// The write through `get_mut`'s entry. `unwrap` is declared over `Option`
/// and over `Result`, so a call of it is a `Signature` decision that waits on
/// its argument's head, and in `get_mut(&mut m, &q).unwrap()` that head is
/// another call's decision; a binding settles the decisions open before it,
/// so `v` is a `&mut i64` where the store and the read are checked. The one
/// spelling still refused binds through a pattern, which lends a shared
/// reference; `docs/std/map.md` carries the same programs.
#[test]
fn a_chained_unwrap_of_get_mut_is_a_mutable_reference_where_it_is_bound() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 1, 10); let q = 1; let o = get_mut(&mut m, &q); \
             let v = o.unwrap(); *v = 99; let r = 1; *get(&m, &r).unwrap()"
        )),
        99
    );
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 1, 10); let q = 1; let v = get_mut(&mut m, &q).unwrap(); \
             *v = 99; let r = 1; *get(&m, &r).unwrap()"
        )),
        99
    );
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 1, 10); let q = 1; let v = get_mut(&mut m, &q).unwrap(); *v"
        )),
        10
    );
    let bound = refusal_at_both(
        &with_map(
            "insert(&mut m, 1, 10); let q = 1; \
             if let Some(v) = get_mut(&mut m, &q) { *v = 99; }; let r = 1; \
             *get(&m, &r).unwrap()",
        ),
        Ty::I64,
    );
    assert!(
        bound.contains("cannot store through `v`, of type &_: not a `&mut`; bind it with `&mut`"),
        "the pattern's binding is a shared reference at the store: {bound}"
    );
}

#[test]
fn or_insert_puts_the_default_once_and_lends_it_after() {
    assert_eq!(
        int(&with_map(
            "let a = or_insert(&mut m, 1, 0); *a = *a + 1; \
             let b = or_insert(&mut m, 1, 0); *b = *b + 1; \
             let c = or_insert(&mut m, 1, 0); *c"
        )),
        2
    );
    assert_eq!(
        count(&with_map(
            "let a = or_insert(&mut m, 1, 0); *a = 1; \
             let b = or_insert(&mut m, 1, 0); *b = 2; len(&m)"
        )),
        1
    );
}

// -- remove, clear ------------------------------------------------------

#[test]
fn remove_answers_the_value_and_drops_the_entry() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 1, 10); let q = 1; remove(&mut m, &q).unwrap()"
        )),
        10
    );
    assert_eq!(
        count(&with_map(
            "insert(&mut m, 1, 10); let q = 1; remove(&mut m, &q); len(&m)"
        )),
        0
    );
}

#[test]
fn remove_of_an_absent_key_is_none_and_changes_nothing() {
    assert_eq!(
        count(&with_map(
            "insert(&mut m, 1, 10); let q = 9; remove(&mut m, &q); len(&m)"
        )),
        1
    );
}

#[test]
fn clear_empties_the_map() {
    assert!(boolean(&with_map(
        "insert(&mut m, 1, 10); insert(&mut m, 2, 20); clear(&mut m); is_empty(&m)"
    )));
}

// -- Iteration order ----------------------------------------------------

#[test]
fn keys_and_values_come_out_in_insertion_order() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 7, 70); insert(&mut m, 3, 30); insert(&mut m, 5, 50); \
             keys(&m) | fold(0, |acc, k| -> acc * 10 + *k)"
        )),
        735
    );
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 7, 70); insert(&mut m, 3, 30); insert(&mut m, 5, 50); \
             values(&m) | fold(0, |acc, v| -> acc * 10 + *v / 10)"
        )),
        735
    );
}

#[test]
fn a_replaced_key_keeps_the_position_it_was_inserted_at() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 7, 70); insert(&mut m, 3, 30); insert(&mut m, 7, 71); \
             keys(&m) | fold(0, |acc, k| -> acc * 10 + *k)"
        )),
        73
    );
}

#[test]
fn a_removed_key_leaves_the_order_of_what_is_left() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 7, 70); insert(&mut m, 3, 30); insert(&mut m, 5, 50); \
             let q = 3; remove(&mut m, &q); \
             keys(&m) | fold(0, |acc, k| -> acc * 10 + *k)"
        )),
        75
    );
}

#[test]
fn into_keys_and_into_values_consume_the_map_in_insertion_order() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 7, 70); insert(&mut m, 3, 30); \
             into_keys(m) | fold(0, |acc, k| -> acc * 10 + k)"
        )),
        73
    );
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 7, 70); insert(&mut m, 3, 30); \
             into_values(m) | fold(0, |acc, v| -> acc * 10 + v / 10)"
        )),
        73
    );
}

// -- extend, retain -----------------------------------------------------

#[test]
fn extend_takes_the_other_map_s_entries_and_its_values_win() {
    assert_eq!(
        count(&format!(
            "let m = hash_map_by({KEYING}); insert(&mut m, 1, 10); \
             let n = hash_map_by({KEYING}); insert(&mut n, 1, 11); insert(&mut n, 2, 20); \
             extend(&mut m, n); len(&m)"
        )),
        2
    );
    assert_eq!(
        int(&format!(
            "let m = hash_map_by({KEYING}); insert(&mut m, 1, 10); \
             let n = hash_map_by({KEYING}); insert(&mut n, 1, 11); \
             extend(&mut m, n); let q = 1; *get(&m, &q).unwrap()"
        )),
        11
    );
}

#[test]
fn retain_keeps_the_entries_the_closure_admits_in_their_order() {
    assert_eq!(
        int(&with_map(
            "insert(&mut m, 7, 70); insert(&mut m, 3, 30); insert(&mut m, 5, 50); \
             retain(&mut m, |k, v| -> *v > 30); \
             keys(&m) | fold(0, |acc, k| -> acc * 10 + *k)"
        )),
        75
    );
}

// -- The loan a reference into the map carries (RFC-0064) ---------------

#[test]
fn an_insert_while_a_get_result_is_live_is_refused() {
    let messages = refusal(
        &with_map(
            "insert(&mut m, 1, 10); let q = 1; let v = get(&m, &q); \
             insert(&mut m, 2, 20); *v.unwrap()",
        ),
        Ty::I64,
    );
    assert!(
        messages.contains("is written here while a reference to it is live"),
        "the borrow check names the conflict: {messages}"
    );
}

// -- Refusals -----------------------------------------------------------

#[test]
fn a_key_of_another_type_is_refused() {
    let messages = refusal(
        &with_map(
            "insert(&mut m, 1, 10); let q = \"x\".to_string(); \
             let found = if let Some(v) = get(&m, &q) { *v } else { 0 }; found",
        ),
        Ty::I64,
    );
    assert!(
        messages.contains("no `get` takes a call of type Fn(&HashMap<i64, i64, Pure>, &String)"),
        "the refusal names the key the map does not hold: {messages}"
    );
}

#[test]
fn a_value_of_another_type_is_refused() {
    let messages = refusal(
        &with_map("insert(&mut m, 1, 10); insert(&mut m, 2, \"x\".to_string()); len(&m)"),
        Ty::U64,
    );
    assert!(
        messages.contains(
            "no `insert` takes a call of type Fn(&mut HashMap<i64, i64, Pure>, i64, String)"
        ),
        "the refusal names the value the map does not hold: {messages}"
    );
}

// -- The set ------------------------------------------------------------

#[test]
fn a_set_that_gained_and_lost_its_one_key_is_empty() {
    assert!(boolean(&with_set(
        "insert(&mut s, 1); let q = 1; remove(&mut s, &q); is_empty(&s)"
    )));
}

#[test]
fn set_insert_answers_whether_the_set_gained_the_key() {
    assert!(boolean(&with_set("insert(&mut s, 1)")));
    assert!(!boolean(&with_set("insert(&mut s, 1); insert(&mut s, 1)")));
    assert_eq!(
        count(&with_set("insert(&mut s, 1); insert(&mut s, 1); len(&s)")),
        1
    );
}

#[test]
fn contains_answers_for_a_key_present_and_one_absent() {
    assert!(boolean(&with_set(
        "insert(&mut s, 1); let q = 1; contains(&s, &q)"
    )));
    assert!(!boolean(&with_set(
        "insert(&mut s, 1); let q = 2; contains(&s, &q)"
    )));
}

#[test]
fn set_remove_answers_whether_the_key_was_there() {
    assert!(boolean(&with_set(
        "insert(&mut s, 1); let q = 1; remove(&mut s, &q)"
    )));
    assert!(!boolean(&with_set(
        "insert(&mut s, 1); let q = 2; remove(&mut s, &q)"
    )));
    assert_eq!(
        count(&with_set(
            "insert(&mut s, 1); let q = 1; remove(&mut s, &q); len(&s)"
        )),
        0
    );
}

#[test]
fn set_clear_empties_the_set() {
    assert!(boolean(&with_set(
        "insert(&mut s, 1); clear(&mut s); is_empty(&s)"
    )));
}

#[test]
fn a_set_is_iterated_in_insertion_order_borrowed_and_consumed() {
    assert_eq!(
        int(&with_set(
            "insert(&mut s, 7); insert(&mut s, 3); \
             as_iter(&s) | fold(0, |acc, k| -> acc * 10 + *k)"
        )),
        73
    );
    assert_eq!(
        int(&with_set(
            "insert(&mut s, 7); insert(&mut s, 3); \
             into_iter(s) | fold(0, |acc, k| -> acc * 10 + k)"
        )),
        73
    );
}

#[test]
fn from_iter_collects_a_sequence_into_a_set_without_its_repeats() {
    assert_eq!(
        count(&format!(
            "let s = range(0, 3) | chain(range(0, 3)) | from_iter({KEYING}); len(&s)"
        )),
        3
    );
}

#[test]
fn set_extend_takes_the_other_set_s_keys() {
    assert_eq!(
        count(&format!(
            "let a = hash_set_by({KEYING}); insert(&mut a, 1); \
             let b = hash_set_by({KEYING}); insert(&mut b, 1); insert(&mut b, 2); \
             extend(&mut a, b); len(&a)"
        )),
        2
    );
    assert_eq!(
        int(&format!(
            "let a = hash_set_by({KEYING}); insert(&mut a, 7); insert(&mut a, 3); \
             let b = hash_set_by({KEYING}); insert(&mut b, 3); insert(&mut b, 5); \
             extend(&mut a, b); into_iter(a) | fold(0, |acc, k| -> acc * 10 + k)"
        )),
        735
    );
}

#[test]
fn intersection_and_difference_each_keep_the_receiver_s_order() {
    let pair = format!(
        "let a = hash_set_by({KEYING}); insert(&mut a, 7); insert(&mut a, 3); \
         let b = hash_set_by({KEYING}); insert(&mut b, 3); insert(&mut b, 5); "
    );
    assert_eq!(
        int(&format!(
            "{pair} into_iter(intersection(a, b)) | fold(0, |acc, k| -> acc * 10 + k)"
        )),
        3
    );
    assert_eq!(
        int(&format!(
            "{pair} into_iter(difference(a, b)) | fold(0, |acc, k| -> acc * 10 + k)"
        )),
        7
    );
}

#[test]
fn is_subset_asks_the_other_set_for_every_key() {
    let pair = format!(
        "let a = hash_set_by({KEYING}); insert(&mut a, 3); \
         let b = hash_set_by({KEYING}); insert(&mut b, 3); insert(&mut b, 5); "
    );
    assert!(boolean(&format!("{pair} is_subset(&a, &b)")));
    assert!(!boolean(&format!("{pair} is_subset(&b, &a)")));
}

#[test]
fn union_takes_both_sets_and_keeps_the_receiver_s_order() {
    let pair = format!(
        "let a = hash_set_by({KEYING}); insert(&mut a, 7); insert(&mut a, 3); \
         let b = hash_set_by({KEYING}); insert(&mut b, 3); insert(&mut b, 5); "
    );
    assert_eq!(
        int(&format!(
            "{pair} into_iter(union(a, b)) | fold(0, |acc, k| -> acc * 10 + k)"
        )),
        735
    );
    assert_eq!(count(&format!("{pair} len(&union(a, b))")), 3);
}

#[test]
fn union_refuses_a_set_of_another_key_type() {
    let messages = refusal_at_both(
        &format!(
            "let a = hash_set_by({KEYING}); insert(&mut a, 7); \
             let b = hash_set_by({KEYING}); insert(&mut b, \"7\".to_string()); \
             len(&union(a, b))"
        ),
        Ty::U64,
    );
    assert!(
        !messages.is_empty(),
        "a union of two sets of different key types compiled"
    );
}
