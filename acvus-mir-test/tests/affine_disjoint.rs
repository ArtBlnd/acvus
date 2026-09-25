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
    // RFC-0098 rule 1: two iterations reach one slot here too, and the
    // updates at a slot combine by `+`, so the storage is keyed by `i / 2`.
    let halved = "let xs = vec([5, 3, 8, 1]); let out = vec([0, 0]);
         for i in 0u64..xs.len() { out[i / 2u64] = out[i / 2u64] + xs[i]; }
         out.len()";
    let orders = storage_orders(halved);
    assert!(
        matches!(&orders[..], [order] if order.starts_with("keyed(")),
        "{}",
        listing(halved)
    );
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

/// W05: `n - 1u64` the body computes from `n` and a word is invariant where
/// it stands (RFC-0066 rule 3), so `(n - 1) - i` is affine and the store
/// is `Disjoint`; both subtractions stay in the body's free stage, and none
/// is hoisted above a loop that may run no iteration.
#[test]
fn a_store_at_an_invariant_the_body_computes_less_the_counter_is_disjoint() {
    let source = "let xs = vec([5, 3, 8, 1]); let n = xs.len(); let out = vec([0, 0, 0, 0]);
         for i in 0u64..n { out[n - 1u64 - i] = xs[i]; }
         out.len()";
    assert_eq!(storage_orders(source), ["disjoint"], "{}", listing(source));
    let free = facts(source)
        .into_iter()
        .find(|fact| fact.contains(": free {"))
        .expect("the free stage");
    let (_, held) = free.split_once('{').expect("a stage lists what it holds");
    let subtractions = held
        .trim_end_matches('}')
        .split(", ")
        .filter(|operation| *operation == "-")
        .count();
    assert_eq!(subtractions, 2, "{free}");
}

/// An element read at a constant index is invariant where the loop does
/// not write its storage, and the store it indexes is `Disjoint`; where the
/// loop writes that storage, it is not, and the store stays in order.
#[test]
fn an_index_read_from_a_storage_the_loop_writes_is_no_invariant() {
    let unwritten = "let xs = vec([5, 3, 8, 1]); let v = vec([2u64, 1u64, 0u64]);
         let out = vec([0, 0, 0]);
         for i in 0u64..3u64 { out[v[0u64] - i] = xs[i]; }
         out.len()";
    assert_eq!(storage_orders(unwritten), ["disjoint"], "{}", listing(unwritten));
    let written = "let v = vec([2u64, 1u64, 0u64]);
         for i in 0u64..3u64 { let c = v[0u64]; v[c - i] = 7u64; }
         v.len()";
    assert_eq!(storage_orders(written), ["in_order"], "{}", listing(written));
}

/// `out.len()` read before the one push every iteration makes is
/// `{len(out) on entry, 1}` (RFC-0066 rule 4, `push`'s
/// `len(c) = old(len(c)) + 1`), so a store at it is `Disjoint`, and the
/// read is `len(out)` above the header `+ k` (rule 7), which leaves the
/// push's stage after the store's; a conditional push, a second push or a
/// pop leaves it in order.
#[test]
fn a_store_at_the_length_an_unconditional_push_grows_is_disjoint() {
    let orders = |body: &str| {
        storage_orders(&format!(
            "let xs = vec([5, 3, 8, 1]); let out = vec([]); let pos = vec([0, 0, 0, 0, 0, 0, 0, 0]);
             for x in &xs {{ {body} }}
             pos.len()"
        ))
    };
    assert_eq!(
        orders("pos[out.len()] = *x; out.push(*x);"),
        ["disjoint", "in_order"],
        "`out` in order through its push, `pos` disjoint"
    );
    for body in [
        "pos[out.len()] = *x; if *x > 2 { out.push(*x); };",
        "pos[out.len()] = *x; out.push(*x); out.push(*x);",
        "pos[out.len()] = *x; out.push(*x); out.pop();",
    ] {
        assert_eq!(orders(body), ["in_order", "in_order"], "{body}");
    }
}

/// RFC-0089 rule 4 asks `a ≠ 0` of the step: an invariant the interval
/// domain proves one nonzero constant is that constant.
#[test]
fn a_stride_bound_through_a_let_is_read_as_its_constant() {
    let source = "let w = 7u64;
         let cur = filled(w * 5u64, 0);
         for r in 0u64..5u64 { cur[r * w] = 100; }
         cur.len()";
    assert_eq!(storage_orders(source), ["disjoint"], "{}", listing(source));
}

/// A step the interval domain puts in `[3, 4]` is nonzero, and bases one
/// apart are no multiple of it.
#[test]
fn a_stride_the_interval_proves_nonzero_is_disjoint_by_its_least_magnitude() {
    let source = "let v = vec([1, 2]);
         let w = if v.len() > 5u64 { 3u64 } else { 4u64 };
         let cur = filled(32u64, 0);
         for r in 0u64..5u64 { cur[r * w] = 100; cur[r * w + 1u64] = 7; }
         cur.len()";
    assert_eq!(storage_orders(source), ["disjoint"], "{}", listing(source));
}

/// Bases three apart meet where the step is 3.
#[test]
fn bases_as_far_apart_as_the_least_stride_stay_in_order() {
    let source = "let v = vec([1, 2]);
         let w = if v.len() > 5u64 { 3u64 } else { 4u64 };
         let cur = filled(32u64, 0);
         for r in 0u64..5u64 { cur[r * w] = 100; cur[r * w + 3u64] = 7; }
         cur.len()";
    assert_eq!(storage_orders(source), ["in_order"], "{}", listing(source));
}

/// A step the interval domain puts in `[0, 4]` may be zero.
#[test]
fn a_stride_that_may_be_zero_stays_in_order() {
    let source = "let v = vec([1, 2]);
         let w = if v.len() > 5u64 { 0u64 } else { 4u64 };
         let cur = filled(32u64, 0);
         for r in 0u64..5u64 { cur[r * w] = 100; }
         cur.len()";
    assert_eq!(storage_orders(source), ["in_order"], "{}", listing(source));
}
