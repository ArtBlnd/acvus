//! RFC-0098 rule 2: a table's token is keyed only at `Equiv`, its entry
//! opened by a call stating `law(absent = v)` whose default is the per-key
//! law's identity, and its keys join in chunk order (rule 3).

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
