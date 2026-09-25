//! A storage a `for` reaches only under one path component that is an
//! affine value of the counter at every access, one step and bases no
//! multiple of it apart, is `Disjoint` (RFC-0089 rule 4), and a store two
//! iterations can both reach stays `InOrder`. A header parameter carrying
//! work on the previous element carries nothing (RFC-0066 rule 7).

use acvus_mir::graph::optimize::Opt;
use acvus_mir::printer::dump_with_facts;
use acvus_mir_test::compile_script_at;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn listing(source: &str) -> String {
    let interner = Interner::new();
    let compiled = compile_script_at(&interner, source, &FxHashMap::default(), Opt::Full)
        .unwrap_or_else(|e| panic!("{source}\n{e}"));
    dump_with_facts(&interner, &compiled.module, &compiled.laws)
}

fn facts(source: &str) -> Vec<String> {
    listing(source)
        .lines()
        .filter_map(|line| line.split_once("// ").map(|(_, fact)| fact.to_string()))
        .filter(|fact| fact.starts_with('L'))
        .collect()
}

/// The order of each storage cycle, outer loop first where loops nest.
fn storage_orders(source: &str) -> Vec<String> {
    facts(source)
        .iter()
        .filter_map(|fact| {
            let (_, cycle) = fact.split_once("cycle Storage(")?;
            let (_, rest) = cycle.split_once(") ")?;
            Some(rest.split_whitespace().next()?.to_string())
        })
        .collect()
}

#[test]
fn a_slot_read_and_written_at_the_counter_is_disjoint_and_its_read_runs_free() {
    let source = "let v = vec([5, 3, 8, 1]);
         for i in 0u64..v.len() { v[i] = v[i] * 2; }
         v.len()";
    assert_eq!(storage_orders(source), ["disjoint"], "{}", listing(source));
    let cycle = facts(source)
        .into_iter()
        .find(|fact| fact.contains("cycle Storage("))
        .expect("the store's cycle");
    assert!(cycle.ends_with("disjoint {index_set}"), "{cycle}");
}

#[test]
fn a_read_of_another_storage_is_no_access_of_the_written_one() {
    let source = "let a = vec([6, 3, 3, 6, 6]);
         let b = vec([0, 0, 0, 0, 0]);
         for i in 1u64..4u64 { b[i] = a[i - 1u64] + a[i] + a[i + 1u64]; }
         b.len()";
    assert_eq!(storage_orders(source), ["disjoint"], "{}", listing(source));
}

#[test]
fn an_index_a_carried_variable_steps_is_read_off_the_counter_and_carries_nothing() {
    let source = "let xs = vec([5, 3, 8]);
         let out = vec([0, 0, 0, 0, 0, 0]);
         let j = 1u64;
         for i in 0u64..xs.len() { out[j] = xs[i]; j = j + 2u64; }
         out.len()";
    assert_eq!(storage_orders(source), ["disjoint"], "{}", listing(source));
    assert!(
        !facts(source).iter().any(|fact| fact.contains("Carried(")),
        "{}",
        listing(source)
    );
}

#[test]
fn a_nested_store_is_disjoint_in_each_loop_by_the_component_of_its_counter() {
    let transpose = "let m = vec([vec([1, 2, 3]), vec([4, 5, 6])]);
         let t = vec([vec([0, 0]), vec([0, 0]), vec([0, 0])]);
         for i in 0u64..2u64 { for j in 0u64..3u64 { t[j][i] = m[i][j]; } }
         t.len()";
    assert_eq!(
        storage_orders(transpose),
        ["disjoint", "disjoint"],
        "{}",
        listing(transpose)
    );
    let row_scale = "let m = vec([vec([1, 2]), vec([3, 4])]);
         for i in 0u64..m.len() {
             for j in 0u64..m[i].len() { m[i][j] = m[i][j] * ((i + 1u64) as i64); }
         }
         m.len()";
    assert_eq!(
        storage_orders(row_scale),
        ["disjoint", "disjoint"],
        "{}",
        listing(row_scale)
    );
}

#[test]
fn a_store_two_iterations_can_both_reach_stays_in_order() {
    let cases = [
        "let xs = vec([5, 3, 8, 1]); let out = vec([0, 0]);
         for i in 0u64..xs.len() { out[i / 2u64] = out[i / 2u64] + xs[i]; }
         out.len()",
        "let xs = vec([5, 3, 8, 1]); let out = vec([0, 0]);
         for i in 0u64..xs.len() { out[0u64] = xs[i]; }
         out.len()",
        "let xs = vec([5, 3, 8, 1]); let out = vec([0, 0]); let c = 1u64;
         for i in 0u64..xs.len() { out[i * 0u64 + c] = xs[i]; }
         out.len()",
        "let xs = vec([5, 3, 8, 1]); let out = vec([0, 0]); let z = (xs[0u64] as u64) - 5u64;
         for i in 0u64..xs.len() { out[i * z] = xs[i]; }
         out.len()",
        "let v = vec([5, 3, 8, 1, 7]);
         for i in 0u64..(v.len() - 1u64) { v[i] = v[i + 1u64]; }
         v.len()",
        "let v = vec([5, 3, 8, 1]);
         for i in 0u64..v.len() { v[i] = (v.len() as i64) + v[i]; }
         v.len()",
        "let v = vec([5, 3, 8, 1]); let c = v.len() - 1u64;
         for i in 1u64..v.len() { v[c - i] = v[c - i + 1u64]; }
         v.len()",
        "let v = vec([5, 3, 8, 1]);
         for i in 0u64..v.len() { v.swap(i, 0u64); }
         v.len()",
        "let v = vec([5, 3, 8, 1, 7, 2]);
         for i in 0u64..2u64 { v.swap(2u64 * i, 2u64 * i + 2u64); }
         v.len()",
    ];
    for source in cases {
        assert_eq!(storage_orders(source), ["in_order"], "{}", listing(source));
    }
    let inner_counter_only = "let out = vec([0, 0, 0]);
         for i in 0u64..3u64 {
             for j in 0u64..3u64 { out[j] = out[j] + (i as i64) + (j as i64); }
         }
         out.len()";
    assert_eq!(
        storage_orders(inner_counter_only),
        ["in_order", "disjoint"],
        "{}",
        listing(inner_counter_only)
    );
}

#[test]
fn an_index_that_counts_down_from_an_invariant_is_disjoint() {
    let reflected = "let xs = vec([5, 3, 8, 1]); let out = vec([0, 0, 0, 0]);
         let c = xs.len() - 1u64;
         for i in 0u64..xs.len() { out[c - i] = xs[i]; }
         out.len()";
    assert_eq!(storage_orders(reflected), ["disjoint"], "{}", listing(reflected));
    let lowered = "let xs = vec([5, 3, 8, 1, 7, 2]); let out = vec([0, 0, 0, 0]);
         for i in 2u64..6u64 { out[i - 2u64] = xs[i]; }
         out.len()";
    assert_eq!(storage_orders(lowered), ["disjoint"], "{}", listing(lowered));
}

#[test]
fn terms_of_one_step_whose_bases_are_no_multiple_of_it_apart_are_disjoint() {
    let pairs = "let v = vec([5, 3, 8, 1]);
         for i in 0u64..(v.len() / 2u64) { v[2u64 * i] = v[2u64 * i + 1u64]; }
         v.len()";
    assert_eq!(storage_orders(pairs), ["disjoint"], "{}", listing(pairs));
}

#[test]
fn a_swap_of_the_two_places_its_declaration_states_is_disjoint() {
    let source = "let v = vec([5, 3, 8, 1]);
         for i in 0u64..(v.len() / 2u64) { v.swap(2u64 * i, 2u64 * i + 1u64); }
         v.len()";
    assert_eq!(storage_orders(source), ["disjoint"], "{}", listing(source));
}

fn carried_count(source: &str) -> usize {
    facts(source)
        .iter()
        .map(|fact| fact.matches("Carried(").count())
        .sum()
}

#[test]
fn a_parameter_carrying_work_on_the_previous_element_carries_nothing() {
    let previous = "let xs = vec([1, 1, 2, 2, 3]); let prev = -1; let runs = 0;
         for x in &xs { if *x != prev { runs = runs + 1; }; prev = *x; }
         runs";
    assert_eq!(carried_count(previous), 1, "{}", listing(previous));
    let class = "let bs = \"a b\".to_string().to_bytes(); let in_word = false; let words = 0;
         for b in &bs {
             let sp = *b == b' ';
             if !sp && !in_word { words = words + 1; };
             in_word = !sp;
         }
         words";
    assert_eq!(carried_count(class), 1, "{}", listing(class));
}

#[test]
fn a_parameter_carrying_what_the_loop_writes_stays_carried() {
    let source = "let xs = vec([1, 1, 2, 2, 3]); let buf = vec([0]); let prev = -1; let runs = 0;
         for x in &xs {
             if *x != prev { runs = runs + 1; };
             prev = buf[0u64];
             buf[0u64] = *x;
         }
         runs";
    assert!(
        facts(source).iter().any(|fact| fact.contains("Carried(") && !fact.contains("any_order")),
        "{}",
        listing(source)
    );
}
