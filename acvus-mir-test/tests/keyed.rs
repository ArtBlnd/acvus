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
    let Order::Keyed { key, within } = c.order else {
        panic!("keyed:\n{}", c.listing)
    };
    assert_eq!(within, KeyOrder::AnyOrder, "{}", c.listing);
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

/// The programs `acvus-interpreter-test`'s soundness harness runs at both
/// levels under `tests/soundness/keyed` exercise a keyed cycle.
#[test]
fn every_keyed_soundness_program_is_keyed() {
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
    assert_eq!(programs.len(), 5, "{programs:?}");
    for program in programs {
        let source = std::fs::read_to_string(&program).expect("the program");
        let c = outer_storage_cycle(&source);
        assert!(
            matches!(c.order, Order::Keyed { .. }),
            "{}:\n{}",
            program.display(),
            c.listing
        );
    }
}
