//! RFC-0098 rules 1 and 3 as `analysis::loop_deps` judges them, and
//! RFC-0089 rule 4's precedence over them.

use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loop_deps::{Head, KeyOrder, Law, LawOp, LoopDeps, Order, Storage, Token};
use acvus_mir::analysis::loops::natural_loops_innermost_first;
use acvus_mir::cfg::{CfgBody, promote};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{BinOp, InstKind, ValueId};
use acvus_mir::printer::dump_with_facts;
use acvus_mir_test::compile_script_at;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

struct StorageCycle {
    listing: String,
    cfg: CfgBody,
    order: Order,
    law: Option<Law>,
}

fn outer_storage_cycle(source: &str) -> StorageCycle {
    outer_storage_cycle_at(source, Opt::Full)
}

fn outer_storage_cycle_at(source: &str, opt: Opt) -> StorageCycle {
    let interner = Interner::new();
    let compiled = compile_script_at(&interner, source, &FxHashMap::default(), opt)
        .unwrap_or_else(|e| panic!("{source}\n{e}"));
    let listing = dump_with_facts(&interner, &compiled.module, &compiled.laws);
    let cfg = promote(compiled.module.main.clone());
    let header = natural_loops_innermost_first(&cfg, &DomTree::build(&cfg))
        .iter()
        .filter(|loop_| Head::of(&cfg.blocks[loop_.header.0].terminator).is_some())
        .max_by_key(|loop_| loop_.block_count())
        .map(|loop_| loop_.header)
        .unwrap_or_else(|| panic!("a staged loop:\n{listing}"));
    let deps = LoopDeps::of(&cfg, &compiled.laws, header)
        .unwrap_or_else(|fault| panic!("{}:\n{listing}", fault.shown()));
    let judged = deps.judge(&cfg, &compiled.laws);
    let found: Vec<(Order, Option<Law>)> = deps
        .cycles
        .iter()
        .zip(judged)
        .filter(|(cycle, _)| matches!(cycle.tokens[..], [Token::Storage(Storage::Slot(_))]))
        .map(|(_, judged)| (judged.order, judged.law.map(|law| law.accumulator.law)))
        .collect();
    let [(order, law)] = &found[..] else {
        panic!("one storage cycle:\n{listing}")
    };
    StorageCycle {
        order: *order,
        law: law.clone(),
        listing,
        cfg,
    }
}

struct KeyedJudgement {
    keyed: bool,
    listing: String,
}

/// Whether any staged loop of the program has a one-storage cycle judged
/// keyed. A program renders its result after its loop, and RFC-0099 fuses
/// that render into a loop of its own, whose one storage is a `push`
/// folded in order.
fn any_keyed_storage_cycle(source: &str) -> KeyedJudgement {
    let interner = Interner::new();
    let compiled = compile_script_at(&interner, source, &FxHashMap::default(), Opt::Full)
        .unwrap_or_else(|e| panic!("{source}\n{e}"));
    let listing = dump_with_facts(&interner, &compiled.module, &compiled.laws);
    let cfg = promote(compiled.module.main.clone());
    let mut keyed = false;
    for loop_ in natural_loops_innermost_first(&cfg, &DomTree::build(&cfg))
        .iter()
        .filter(|loop_| Head::of(&cfg.blocks[loop_.header.0].terminator).is_some())
    {
        let deps = LoopDeps::of(&cfg, &compiled.laws, loop_.header)
            .unwrap_or_else(|fault| panic!("{}:\n{listing}", fault.shown()));
        let judged = deps.judge(&cfg, &compiled.laws);
        keyed |= deps.cycles.iter().zip(judged).any(|(cycle, judged)| {
            matches!(cycle.tokens[..], [Token::Storage(Storage::Slot(_))])
                && matches!(judged.order, Order::Keyed { .. })
        });
    }
    KeyedJudgement { keyed, listing }
}

fn is_remainder(cfg: &CfgBody, value: ValueId) -> bool {
    cfg.blocks.iter().flat_map(|block| &block.insts).any(|inst| {
        matches!(inst.kind, InstKind::BinOp { dst, op: BinOp::Mod, .. } if dst == value)
    })
}

/// Corpus row K02.
#[test]
fn a_histogram_bucket_updated_by_add_is_keyed_any_order_at_its_bucket() {
    let c = outer_storage_cycle(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0]);
         for x in &xs { let b = *x % 4u64; h[b] = h[b] + 1; }
         h.len()",
    );
    let Order::Keyed {
        key,
        within,
        across,
    } = c.order
    else {
        panic!("keyed:\n{}", c.listing)
    };
    assert_eq!(within, KeyOrder::AnyOrder, "{}", c.listing);
    assert_eq!(across, KeyOrder::AnyOrder, "a vec's buckets join in any order:\n{}", c.listing);
    assert!(is_remainder(&c.cfg, key), "the key is `*x % 4`:\n{}", c.listing);
    assert_eq!(c.law, Some(Law::Op(LawOp::Add)), "{}", c.listing);
    assert!(
        c.listing
            .contains("any_order law(Op(Add) exact commutative) {index, +, index_set}"),
        "{}",
        c.listing
    );
}

/// RFC-0098 rule 3.
#[test]
fn a_bucket_overwritten_under_a_branch_is_keyed_last_in_order_within_its_key() {
    let c = outer_storage_cycle(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0u64, 0u64, 0u64, 0u64]);
         for x in &xs {
             let b = *x % 4u64;
             let old = h[b];
             h[b] = if *x > 2u64 { *x } else { old };
         }
         h.len()",
    );
    let Order::Keyed { within, .. } = c.order else {
        panic!("keyed:\n{}", c.listing)
    };
    assert_eq!(within, KeyOrder::InOrder, "{}", c.listing);
    assert_eq!(c.law, Some(Law::Last), "{}", c.listing);
}

#[test]
fn a_bucket_read_at_another_index_than_it_is_written_is_no_keyed_storage() {
    let c = outer_storage_cycle(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0]);
         for x in &xs { let b = *x % 4u64; let c = (*x + 1u64) % 4u64; h[b] = h[c] + 1; }
         h.len()",
    );
    assert_eq!((c.order, c.law), (Order::InOrder, None), "{}", c.listing);
}

#[test]
fn a_histogram_that_reads_its_length_in_the_loop_is_no_keyed_storage() {
    let c = outer_storage_cycle(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0u64, 0u64, 0u64, 0u64]);
         for x in &xs { let b = *x % 4u64; h[b] = h[b] + h.len(); }
         h.len()",
    );
    assert_eq!((c.order, c.law), (Order::InOrder, None), "{}", c.listing);
}

/// The inner loop reads no element, so only its traversal reaches the
/// histogram. The full pipeline refuses this program with "For takes &[_]
/// and got &mut [u64]", as it already did at 20133ee6, so it is judged as
/// `--opt none` lowers it.
#[test]
fn a_histogram_that_iterates_over_itself_in_the_loop_is_no_keyed_storage() {
    let c = outer_storage_cycle_at(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0u64, 0u64, 0u64, 0u64]);
         let seen = 0u64;
         for x in &xs { let b = *x % 4u64; h[b] = h[b] + 1u64; for _y in &h { seen = seen + 1u64; } }
         seen",
        Opt::None,
    );
    assert_eq!((c.order, c.law), (Order::InOrder, None), "{}", c.listing);
}

/// The reference `len` reads is an `Option`'s payload, which names no place
/// of the histogram, so the calls through it reach all of it.
#[test]
fn a_histogram_lent_through_an_option_payload_is_no_keyed_storage() {
    let c = outer_storage_cycle(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0u64, 0u64, 0u64, 0u64]);
         let n = 0u64;
         for x in &xs {
             let b = *x % 4u64;
             h[b] = h[b] + 1u64;
             let r = Some(&h).unwrap();
             n = n + r.len();
         }
         n",
    );
    assert_eq!((c.order, c.law), (Order::InOrder, None), "{}", c.listing);
}

#[test]
fn a_bucket_updated_without_a_law_is_no_keyed_storage() {
    let c = outer_storage_cycle(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([9, 9, 9, 9]);
         for x in &xs { let b = *x % 4u64; h[b] = h[b] / 2; }
         h.len()",
    );
    assert_eq!((c.order, c.law), (Order::InOrder, None), "{}", c.listing);
}

/// RFC-0089 rule 4 before RFC-0098 rule 1.
#[test]
fn an_element_updated_at_the_counter_stays_disjoint() {
    let c = outer_storage_cycle(
        "let h = vec([1, 2, 3, 4]);
         for i in 0u64..4u64 { h[i] = h[i] + 1; }
         h.len()",
    );
    assert_eq!((c.order, c.law), (Order::Disjoint, None), "{}", c.listing);
}

// -- RFC-0098 rule 1: two key values the program shows equal are one key --

const LEVELS: [Opt; 2] = [Opt::None, Opt::Full];

fn keyed_at_every_level(source: &str) {
    for opt in LEVELS {
        let c = outer_storage_cycle_at(source, opt);
        let Order::Keyed { within, .. } = c.order else {
            panic!("keyed at {opt:?}:\n{}", c.listing)
        };
        assert_eq!(within, KeyOrder::AnyOrder, "{opt:?}:\n{}", c.listing);
        assert_eq!(c.law, Some(Law::Op(LawOp::Add)), "{opt:?}:\n{}", c.listing);
    }
}

fn in_order_at_every_level(source: &str) {
    for opt in LEVELS {
        let c = outer_storage_cycle_at(source, opt);
        assert_eq!((c.order, c.law), (Order::InOrder, None), "{opt:?}:\n{}", c.listing);
    }
}

/// App corpus rows 01:15 and 02:17: `*e` read twice through the element's
/// `&u64` is one key.
#[test]
fn a_degree_count_reading_its_key_twice_through_one_shared_reference_is_keyed() {
    keyed_at_every_level(
        "let es = vec([0u64, 1u64, 2u64, 2u64, 3u64]);
         let degree = filled(4u64, 0);
         for e in &es { degree[*e] = degree[*e] + 1; }
         degree.len()",
    );
}

/// `*x % 4u64` computed twice: one `%` over a copy through one shared
/// reference and one constant.
#[test]
fn a_bucket_computed_twice_by_one_remainder_is_keyed() {
    keyed_at_every_level(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0]);
         for x in &xs { h[*x % 4u64] = h[*x % 4u64] + 1; }
         h.len()",
    );
}

/// The two divisions have one operand pair, so the first traps wherever
/// the second would, before any access to `h`.
#[test]
fn a_bucket_computed_twice_by_one_division_by_a_divisor_of_the_iteration_is_keyed() {
    keyed_at_every_level(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0, 0, 0, 0, 0, 0]);
         for x in &xs { let y = *x % 3u64 + 1u64; h[*x / y] = h[*x / y] + 1; }
         h.len()",
    );
}

/// The element's own reference and `&xs[i]` name one element, but they are
/// two references, and the program does not show them equal.
#[test]
fn a_key_read_through_two_references_to_one_element_is_no_keyed_storage() {
    in_order_at_every_level(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0, 0, 0, 0, 0, 0]);
         let i = 0u64;
         for x in &xs { let q = &xs[i]; h[*q] = h[*x] + 1; i = i + 1u64; }
         h.len()",
    );
}

#[test]
fn a_key_read_twice_through_a_mutable_reference_is_no_keyed_storage() {
    in_order_at_every_level(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0, 0, 0, 0, 0, 0]);
         for x in &mut xs { h[*x] = h[*x] + 1; }
         h.len()",
    );
}

/// `wrapping_add` is a call, so its two results are two keys.
#[test]
fn a_key_computed_twice_by_a_call_is_no_keyed_storage() {
    in_order_at_every_level(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0]);
         for x in &xs { h[wrapping_add(*x, 1u64) % 4u64] = h[wrapping_add(*x, 1u64) % 4u64] + 1; }
         h.len()",
    );
}

/// `k` is lent, so it stays a place at both levels: read, written through
/// `r`, and read again.
#[test]
fn a_key_read_from_a_place_written_between_the_reads_is_no_keyed_storage() {
    in_order_at_every_level(
        "let xs = vec([4u64, 1u64, 8u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0]);
         for x in &xs {
             let k = *x % 4u64;
             let a = h[k];
             let r = &mut k;
             *r = (*r + 1u64) % 4u64;
             h[k] = a + 1;
         }
         h.len()",
    );
}

#[test]
fn a_key_and_the_key_plus_one_are_no_keyed_storage() {
    in_order_at_every_level(
        "let xs = vec([4u64, 1u64, 5u64, 3u64, 6u64, 7u64]);
         let h = vec([0, 0, 0, 0, 0, 0, 0, 0, 0]);
         for x in &xs { h[*x] = h[*x + 1u64] + 1; }
         h.len()",
    );
}

#[test]
fn a_key_plus_one_and_the_key_plus_two_are_no_keyed_storage() {
    in_order_at_every_level(
        "let xs = vec([4u64, 1u64, 5u64, 3u64, 6u64, 0u64]);
         let h = vec([0, 0, 0, 0, 0, 0, 0, 0, 0]);
         for x in &xs { h[*x + 1u64] = h[*x + 2u64] + 1; }
         h.len()",
    );
}

#[test]
fn a_key_plus_one_and_the_key_times_one_are_no_keyed_storage() {
    in_order_at_every_level(
        "let xs = vec([4u64, 1u64, 5u64, 3u64, 6u64, 0u64]);
         let h = vec([0, 0, 0, 0, 0, 0, 0, 0, 0]);
         for x in &xs { h[*x + 1u64] = h[*x * 1u64] + 1; }
         h.len()",
    );
}

#[test]
fn a_remainder_and_the_remainder_of_its_operands_swapped_are_no_keyed_storage() {
    in_order_at_every_level(
        "let xs = vec([5u64, 3u64, 6u64]);
         let h = vec([0, 0, 0, 0, 0, 0, 0, 0]);
         for x in &xs { h[*x % 7u64] = h[7u64 % *x] + 1; }
         h.len()",
    );
}

/// `seen_vec` (remaining-rows survey): RFC-0093 rule 5's unconditional
/// store of a value reading none of the bucket is `last`, which keeps each
/// key's stores in order.
#[test]
fn a_bucket_stored_a_value_reading_none_of_it_on_every_path_is_keyed_last() {
    let c = outer_storage_cycle(
        "let xs = vec([4u64, 1u64, 4u64, 3u64, 1u64, 4u64]);
         let seen = vec([false, false, false, false, false]);
         for x in &xs { seen[*x] = true; }
         seen.len()",
    );
    let Order::Keyed { within, across, .. } = c.order else {
        panic!("keyed:\n{}", c.listing)
    };
    assert_eq!(within, KeyOrder::InOrder, "{}", c.listing);
    assert_eq!(across, KeyOrder::AnyOrder, "{}", c.listing);
    assert_eq!(c.law, Some(Law::Last), "{}", c.listing);
    assert!(c.listing.contains("in_order law(Last exact) {index_set}"), "{}", c.listing);
}

/// Corpus row W06's scatter, with a repeated target: `last` per key holds
/// whether or not `p` is injective, since a key's stores keep their order.
#[test]
fn a_scatter_through_an_index_vector_is_keyed_last_whether_or_not_it_repeats_a_key() {
    let c = outer_storage_cycle(
        "let xs = vec([5, 3, 8, 1]);
         let p = vec([1u64, 3u64, 1u64, 2u64]);
         let out = vec([0, 0, 0, 0]);
         for i in 0u64..xs.len() { out[p[i]] = xs[i]; }
         out.len()",
    );
    assert!(matches!(c.order, Order::Keyed { within: KeyOrder::InOrder, .. }), "{}", c.listing);
    assert_eq!(c.law, Some(Law::Last), "{}", c.listing);
}

/// The storage cycles of the loop holding every other one.
fn outer_storage_cycles(source: &str) -> (Vec<(Order, Option<Law>)>, String) {
    let interner = Interner::new();
    let compiled = compile_script_at(&interner, source, &FxHashMap::default(), Opt::Full)
        .unwrap_or_else(|e| panic!("{source}\n{e}"));
    let listing = dump_with_facts(&interner, &compiled.module, &compiled.laws);
    let cfg = promote(compiled.module.main.clone());
    let header = natural_loops_innermost_first(&cfg, &DomTree::build(&cfg))
        .iter()
        .filter(|loop_| Head::of(&cfg.blocks[loop_.header.0].terminator).is_some())
        .max_by_key(|loop_| loop_.block_count())
        .map(|loop_| loop_.header)
        .unwrap_or_else(|| panic!("a staged loop:\n{listing}"));
    let deps = LoopDeps::of(&cfg, &compiled.laws, header)
        .unwrap_or_else(|fault| panic!("{}:\n{listing}", fault.shown()));
    let judged = deps.judge(&cfg, &compiled.laws);
    let found = deps
        .cycles
        .iter()
        .zip(judged)
        .filter(|(cycle, _)| matches!(cycle.tokens[..], [Token::Storage(Storage::Slot(_))]))
        .map(|(_, judged)| (judged.order, judged.law.map(|law| law.accumulator.law)))
        .collect();
    (found, listing)
}

/// `wc_lines`: RFC-0098 rule 1's nested reading. The outer iteration
/// touches the table only inside the word loop, whose own cycle on it is
/// keyed by `+`.
#[test]
fn a_word_count_over_lines_is_keyed_at_the_outer_loop_by_its_word_loops_law() {
    let c = outer_storage_cycle(
        "let lines = vec([vec([\"x\".to_string(), \"y\".to_string()]), vec([\"z\".to_string()]), vec([\"x\".to_string(), \"z\".to_string()])]);
         let m = hash_map();
         for i in 0u64..lines.len() {
             for w in &lines[i] {
                 let e = or_insert(&mut m, w.to_string(), 0);
                 *e = *e + 1;
             }
         }
         (*get(&m, &\"x\".to_string()).unwrap()).to_string()",
    );
    assert_eq!(
        c.order,
        match c.order {
            Order::Keyed { key, .. } => Order::Keyed {
                key,
                within: KeyOrder::AnyOrder,
                across: KeyOrder::InOrder,
            },
            _ => panic!("keyed:\n{}", c.listing),
        },
        "{}",
        c.listing
    );
    assert_eq!(c.law, Some(Law::Op(LawOp::Add)), "{}", c.listing);
    let keyed_lines = c
        .listing
        .lines()
        .filter(|line| line.contains("keys in_order) any_order law(Op(Add) exact commutative)"))
        .count();
    assert_eq!(keyed_lines, 2, "the word loop and the line loop:\n{}", c.listing);
}

#[test]
fn a_histogram_over_rows_is_keyed_at_the_outer_loop() {
    let c = outer_storage_cycle(
        "let rows = vec([vec([4u64, 1u64]), vec([8u64]), vec([3u64, 6u64, 7u64])]);
         let h = vec([0, 0, 0, 0]);
         for i in 0u64..rows.len() {
             for x in &rows[i] { let b = *x % 4u64; h[b] = h[b] + 1; }
         }
         h[0]",
    );
    assert!(
        matches!(c.order, Order::Keyed { within: KeyOrder::AnyOrder, across: KeyOrder::AnyOrder, .. }),
        "{}",
        c.listing
    );
    assert_eq!(c.law, Some(Law::Op(LawOp::Add)), "{}", c.listing);
}

/// `adv_wc_len` and its kin: the outer iteration reads the table outside
/// the word loop (its length, a lookup), or touches it in two nested
/// loops. A split of the outer loop would hand those reads a chunk's own
/// table.
#[test]
fn a_table_the_outer_iteration_reads_outside_one_keyed_nested_loop_is_not_keyed_there() {
    let lines = "let lines = vec([vec([\"x\".to_string(), \"y\".to_string()]), vec([\"z\".to_string()]), vec([\"x\".to_string(), \"z\".to_string()])]);";
    let count = "for w in &lines[i] { let e = or_insert(&mut m, w.to_string(), 0); *e = *e + 1; }";
    for (before, after, result) in [
        ("let sizes = vec([]);", "sizes.push(m.len());", "sizes.len()"),
        ("let seen = 0;", "seen = seen + *get(&m, &\"x\".to_string()).unwrap_or(&0);", "seen"),
        ("", count, "m.len()"),
    ] {
        let (cycles, listing) = outer_storage_cycles(&format!(
            "{lines} let m = hash_map(); {before}
             for i in 0u64..lines.len() {{ {count} {after} }}
             {result}"
        ));
        assert!(
            cycles.iter().all(|(order, _)| !matches!(order, Order::Keyed { .. })),
            "{after}:\n{listing}"
        );
    }
}

/// The programs under `acvus-interpreter-test`'s `tests/soundness/keyed`,
/// which its soundness harness runs at both levels, and whether each is a
/// keyed cycle.
const KEYED_SOUNDNESS_PROGRAMS: &[(&str, bool)] = &[
    ("degree_count_two_takes.acvus", true),
    ("division_by_zero_in_the_key_traps.acvus", true),
    ("k02_histogram_vec_bucket.acvus", true),
    ("k07_char_frequency.acvus", true),
    ("last_per_key_in_order.acvus", true),
    ("map_count_by_lambdas.acvus", false),
    ("map_count_from_five.acvus", false),
    ("map_count_over_a_filled_map.acvus", true),
    ("map_group_by_push_from_new.acvus", true),
    ("map_max_per_key.acvus", true),
    ("map_word_count_or_insert.acvus", true),
    ("marks_stored_last.acvus", true),
    ("nested_histogram_over_rows.acvus", true),
    // The outer loop of these two is refused (tested by
    // `a_table_the_outer_iteration_reads_outside_one_keyed_nested_loop_is_not_keyed_there`);
    // each word loop inside is itself a keyed count.
    ("nested_word_count_in_two_loops.acvus", true),
    ("nested_word_count_over_lines.acvus", true),
    ("nested_word_count_read_by_lookup.acvus", true),
    ("scatter_repeating_a_key.acvus", true),
    ("two_references_to_one_element.acvus", false),
    ("u8_bucket_past_255.acvus", true),
    ("u8_bucket_past_255_two_takes.acvus", true),
    ("u8_bucket_to_255.acvus", true),
];

#[test]
fn every_keyed_soundness_program_is_judged_as_listed() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lies in the workspace")
        .join("acvus-interpreter-test/tests/soundness/keyed");
    let mut programs: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("the keyed soundness programs")
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "acvus"))
        .collect();
    programs.sort();
    let names: Vec<&str> = programs
        .iter()
        .map(|path| path.file_name().and_then(|name| name.to_str()).expect("a named program"))
        .collect();
    let listed: Vec<&str> = KEYED_SOUNDNESS_PROGRAMS.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, listed);
    for (program, &(_, keyed)) in programs.iter().zip(KEYED_SOUNDNESS_PROGRAMS) {
        let source = std::fs::read_to_string(program).expect("the program");
        let judged = any_keyed_storage_cycle(&source);
        assert_eq!(
            judged.keyed,
            keyed,
            "{}:\n{}",
            program.display(),
            judged.listing
        );
    }
}
