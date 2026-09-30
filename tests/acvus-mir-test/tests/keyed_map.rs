//! RFC-0098 rule 2: a table's token is keyed only at `Equiv`, its entry
//! reached by calls whose terms (RFC-0104) say what each does to it (rule
//! 5), read where the table held none at the per-key law's identity, and
//! its keys join in chunk order (rule 3).

use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loop_deps::{Head, KeyOrder, Law, LawOp, LoopDeps, Order, Storage, Token};
use acvus_mir::analysis::loops::natural_loops_innermost_first;
use acvus_mir::cfg::promote;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::printer::dump_with_facts;
use acvus_mir_test::compile_script_at;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

struct Judgment {
    order: Order,
    law: Option<Law>,
}

#[derive(Debug, PartialEq, Eq)]
struct KeyOrders {
    within: KeyOrder,
    across: KeyOrder,
}

impl KeyOrders {
    fn of(order: &Order) -> Option<KeyOrders> {
        match order {
            Order::Keyed { within, across, .. } => Some(KeyOrders {
                within: *within,
                across: *across,
            }),
            Order::Disjoint | Order::AnyOrder | Order::InOrder => None,
        }
    }
}

struct TableCycle {
    listing: String,
    judgment: Judgment,
}

fn innermost_storage_cycle_at(source: &str, opt: Opt) -> TableCycle {
    let interner = Interner::new();
    let compiled = compile_script_at(&interner, source, &FxHashMap::default(), opt)
        .unwrap_or_else(|e| panic!("{source}\n{e}"));
    let listing = dump_with_facts(&interner, &compiled.module, &compiled.laws);
    let cfg = promote(compiled.module.main.clone());
    let header = natural_loops_innermost_first(&cfg, &DomTree::build(&cfg))
        .iter()
        .find(|loop_| Head::of(&cfg.blocks[loop_.header.0].terminator).is_some())
        .map(|loop_| loop_.header)
        .unwrap_or_else(|| panic!("a staged loop:\n{listing}"));
    let deps = LoopDeps::of(&cfg, &compiled.laws, header)
        .unwrap_or_else(|fault| panic!("{}:\n{listing}", fault.shown()));
    let judged = deps.judge(&cfg, &compiled.laws);
    let mut found: Vec<Judgment> = deps
        .cycles
        .iter()
        .zip(judged)
        .filter(|(cycle, _)| matches!(cycle.tokens[..], [Token::Storage(Storage::Slot(_))]))
        .map(|(_, judged)| Judgment {
            order: judged.order,
            law: judged.law.map(|law| law.accumulator.law),
        })
        .collect();
    let (Some(judgment), None) = (found.pop(), found.pop()) else {
        panic!("one storage cycle:\n{listing}")
    };
    TableCycle { listing, judgment }
}

fn at_both_levels(source: &str) -> TableCycle {
    let full = innermost_storage_cycle_at(source, Opt::Full);
    let none = innermost_storage_cycle_at(source, Opt::None);
    assert_eq!(
        KeyOrders::of(&full.judgment.order),
        KeyOrders::of(&none.judgment.order),
        "{}\n{}",
        full.listing,
        none.listing
    );
    assert_eq!(
        full.judgment.law.as_ref().map(std::mem::discriminant),
        none.judgment.law.as_ref().map(std::mem::discriminant),
        "{}\n{}",
        full.listing,
        none.listing
    );
    full
}

const WORDS: &str = "let words = vec([\"a\".to_string(), \"b\".to_string(), \"a\".to_string()]);";

fn assert_not_keyed(c: &TableCycle) {
    assert_eq!(KeyOrders::of(&c.judgment.order), None, "{}", c.listing);
}

/// Corpus row K01.
#[test]
fn a_word_count_by_or_insert_at_equiv_is_keyed_by_add() {
    let c = at_both_levels(&format!(
        "{WORDS} let m = hash_map();
         for w in &words {{ let n = or_insert(&mut m, w.to_string(), 0); *n = *n + 1; }}
         len(&m)"
    ));
    assert_eq!(
        KeyOrders::of(&c.judgment.order),
        Some(KeyOrders {
            within: KeyOrder::AnyOrder,
            across: KeyOrder::InOrder,
        }),
        "{}",
        c.listing
    );
    assert_eq!(c.judgment.law, Some(Law::Op(LawOp::Add)), "{}", c.listing);
    assert!(
        c.listing.contains("keys in_order) any_order law(Op(Add) exact commutative)"),
        "{}",
        c.listing
    );
}

/// Corpus row K03 opened at `new()`, the identity `push`'s `fold` names.
#[test]
fn a_group_by_push_opened_at_new_is_keyed_by_the_fold_in_order() {
    let c = at_both_levels(&format!(
        "{WORDS} let g = hash_map();
         for w in &words {{ let bucket = or_insert(&mut g, w.to_string(), new()); bucket.push(w.to_string()); }}
         len(&g)"
    ));
    assert_eq!(
        KeyOrders::of(&c.judgment.order),
        Some(KeyOrders {
            within: KeyOrder::InOrder,
            across: KeyOrder::InOrder,
        }),
        "a push per key keeps input order within the key:\n{}",
        c.listing
    );
    assert!(matches!(c.judgment.law, Some(Law::Fold(_))), "{}", c.listing);
}

#[test]
fn the_same_count_over_hash_map_by_is_opaque_and_not_keyed() {
    assert_not_keyed(&at_both_levels(&format!(
        "{WORDS} let m = hash_map_by(|k| -> hash(k), |a, b| -> a == b);
         for w in &words {{ let n = or_insert(&mut m, w.to_string(), 0); *n = *n + 1; }}
         len(&m)"
    )));
}

#[test]
fn a_key_made_by_a_call_twice_in_one_iteration_is_two_keys() {
    assert_not_keyed(&at_both_levels(&format!(
        "{WORDS} let m = hash_map();
         for w in &words {{
             let n = or_insert(&mut m, w.to_string(), 0); *n = *n + 1;
             let o = or_insert(&mut m, w.to_string(), 0); *o = *o + 1;
         }}
         len(&m)"
    )));
}

#[test]
fn two_different_keys_in_one_iteration_are_not_keyed() {
    assert_not_keyed(&at_both_levels(&format!(
        "{WORDS} let m = hash_map();
         for w in &words {{
             let n = or_insert(&mut m, w.to_string(), 0); *n = *n + 1;
             let o = or_insert(&mut m, \"total\".to_string(), 0); *o = *o + 1;
         }}
         len(&m)"
    )));
}

#[test]
fn a_length_read_in_the_loop_makes_the_table_an_ordinary_token() {
    assert_not_keyed(&at_both_levels(&format!(
        "{WORDS} let m = hash_map(); let seen = 0u64;
         for w in &words {{
             let n = or_insert(&mut m, w.to_string(), 0); *n = *n + 1;
             seen = seen + len(&m);
         }}
         seen"
    )));
}

/// RFC-0098 rule 4: a chunk builds its entries from the law's identity, so
/// a default of `5` would be counted once per chunk.
#[test]
fn an_entry_opened_at_another_value_than_the_identity_is_not_keyed() {
    assert_not_keyed(&at_both_levels(&format!(
        "{WORDS} let m = hash_map();
         for w in &words {{ let n = or_insert(&mut m, w.to_string(), 5); *n = *n + 1; }}
         len(&m)"
    )));
}

/// `k03_vec1` (remaining-rows survey): a group-by opened at `vec(["x"])`
/// holds `x` before the first push, which a chunk run from `push`'s
/// identity would drop from every chunk after the first.
#[test]
fn a_group_by_push_opened_at_a_non_empty_vec_is_not_keyed() {
    assert_not_keyed(&at_both_levels(&format!(
        "{WORDS} let g = hash_map();
         for w in &words {{ let bucket = or_insert(&mut g, w.to_string(), vec([\"x\".to_string()])); bucket.push(w.to_string()); }}
         len(&g)"
    )));
}

// -- Entries a call's term reaches (RFC-0104, RFC-0098 rule 5) -------------

/// Corpus row K03 as written: the entry opens at `vec([])`, which is
/// `new()` because `new`'s term is `vec([])` (RFC-0104 rule 3).
#[test]
fn a_group_by_push_opened_at_an_empty_vec_is_keyed_by_the_fold_in_order() {
    let c = at_both_levels(&format!(
        "{WORDS} let g = hash_map();
         for w in &words {{ let bucket = or_insert(&mut g, w.to_string(), vec([])); bucket.push(w.to_string()); }}
         len(&g)"
    ));
    assert_eq!(
        KeyOrders::of(&c.judgment.order),
        Some(KeyOrders {
            within: KeyOrder::InOrder,
            across: KeyOrder::InOrder,
        }),
        "{}",
        c.listing
    );
    assert!(matches!(c.judgment.law, Some(Law::Fold(_))), "{}", c.listing);
}

/// Corpus row K04: a set's `insert` stores `true` whatever the entry held,
/// and its result is dropped, so each key's law is `last`.
#[test]
fn a_dedup_by_set_insert_is_keyed_last_in_order() {
    let c = at_both_levels(
        "let xs = vec([4, 1, 4, 3]); let s = hash_set();
         for x in &xs { s.insert(*x); }
         s.len()",
    );
    assert_eq!(
        KeyOrders::of(&c.judgment.order),
        Some(KeyOrders {
            within: KeyOrder::InOrder,
            across: KeyOrder::InOrder,
        }),
        "{}",
        c.listing
    );
    assert_eq!(c.judgment.law, Some(Law::Last), "{}", c.listing);
}

/// The survey's `adv_k04_first`: the result of `insert` says whether the
/// key was absent, which a chunk would answer once per chunk.
#[test]
fn a_set_insert_whose_result_is_read_is_not_keyed() {
    assert_not_keyed(&at_both_levels(
        "let xs = vec([4, 1, 4, 3]); let s = hash_set(); let n = 0;
         for x in &xs { if s.insert(*x) { n = n + 1; }; }
         n",
    ));
}

/// Corpus row K06: `get` views the entry at `r.k`, and `insert` stores at
/// `to_string(&r.k)`, the same key by `to_string`'s term `*a` (RFC-0098
/// rule 1); the view's `None` arm reads the entry at `0`, `+`'s identity.
#[test]
fn a_sum_by_get_then_insert_is_keyed_by_add() {
    let c = at_both_levels(
        "let rows = vec([{ k: \"a\".to_string(), v: 3, }, { k: \"b\".to_string(), v: 7, }]);
         let m = hash_map();
         for r in &rows {
             let old = match get(&m, &r.k) { Some(v) => *v, None => 0, };
             insert(&mut m, r.k.to_string(), old + r.v);
         }
         len(&m)",
    );
    assert_eq!(
        KeyOrders::of(&c.judgment.order),
        Some(KeyOrders {
            within: KeyOrder::AnyOrder,
            across: KeyOrder::InOrder,
        }),
        "{}",
        c.listing
    );
    assert_eq!(c.judgment.law, Some(Law::Op(LawOp::Add)), "{}", c.listing);
}

/// The survey's `adv_k06_twokeys`: `r.k` and `r.j` are two places.
#[test]
fn a_get_and_an_insert_at_two_fields_are_not_keyed() {
    assert_not_keyed(&at_both_levels(
        "let rows = vec([{ k: \"a\".to_string(), j: \"b\".to_string(), v: 3, }]);
         let m = hash_map();
         for r in &rows {
             let old = match get(&m, &r.k) { Some(v) => *v, None => 0, };
             insert(&mut m, r.j.to_string(), old + r.v);
         }
         len(&m)",
    ));
}

/// A default of `5` in the view's `None` arm is no identity of `+`.
#[test]
fn a_get_read_at_another_value_than_the_identity_is_not_keyed() {
    assert_not_keyed(&at_both_levels(
        "let rows = vec([{ k: \"a\".to_string(), v: 3, }]);
         let m = hash_map();
         for r in &rows {
             let old = match get(&m, &r.k) { Some(v) => *v, None => 5, };
             insert(&mut m, r.k.to_string(), old + r.v);
         }
         len(&m)",
    ));
}

/// The old value `insert` returns, read in the iteration, is the entry's
/// state outside its cycle.
#[test]
fn an_insert_whose_old_value_is_read_is_not_keyed() {
    assert_not_keyed(&at_both_levels(&format!(
        "{WORDS} let m = hash_map(); let again = 0;
         for w in &words {{ let old = insert(&mut m, w.to_string(), 1); if old.is_some() {{ again = again + 1; }}; }}
         again"
    )));
}

/// Corpus row K09: `contains` reads the entry, and the count reads what it
/// read.
#[test]
fn a_count_distinct_by_contains_then_insert_is_not_keyed() {
    let interner = Interner::new();
    let compiled = compile_script_at(
        &interner,
        "let xs = vec([4, 1, 4, 3]); let s = hash_set(); let distinct = 0;
         for x in &xs { if !s.contains(x) { s.insert(*x); distinct = distinct + 1; }; }
         distinct",
        &FxHashMap::default(),
        Opt::Full,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let listing = dump_with_facts(&interner, &compiled.module, &compiled.laws);
    assert!(!listing.contains("keyed("), "{listing}");
}
