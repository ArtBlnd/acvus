//! Each clause of RFC-0066 rule 8 at the listing `acvus mir` prints: the
//! cost line under a `For`, computed from `analysis::loop_deps`' stages
//! that run apart, free or `Disjoint`, and a table. The table here gives each family a weight no sum of
//! the others reaches, so a line's `W` says which rows it counted.

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::compile_source_with_externs;
use acvus_mir::analysis::cost::CostTable;
use acvus_mir::graph::ParsedAst;
use acvus_mir::printer::{dump_with_costs, dump_with_facts};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

const TABLE: CostTable = CostTable {
    arithmetic: 1,
    compare: 1,
    load: 1,
    store: 1,
    allocation: 1,
    local_call: 1,
    extern_call: 10,
    heavy: 100_000,
    spawn: 1_000,
    merge: 1,
    chunk_dispatch: 1_000,
    buffered_element: 1,
    k: 32,
};

const WEIGHED: u64 = 5_000;

#[extern_fn(effect = pure, cost = 5000)]
fn weighed(x: i64) -> i64 {
    x + 1
}

#[extern_fn(effect = pure)]
fn unweighed(x: i64) -> i64 {
    x + 1
}

#[extern_fn(heavy, effect = pure)]
fn offloaded(x: i64) -> i64 {
    x + 1
}

mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "c",
        fn scaled<T>(a: T) -> T
        where
            T: Var<kind::Type>;
    }
}

const SCALED_AT_U64: u64 = 7;

#[extern_fn(instance_of = sig::scaled, effect = pure, cost = 5000)]
fn scaled_i64(a: i64) -> i64 {
    a * 2
}

#[extern_fn(instance_of = sig::scaled, effect = pure, cost = 7)]
fn scaled_u64(a: u64) -> u64 {
    a * 2
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "c",
        signatures: [sig::scaled],
        fns: [weighed, unweighed, offloaded, scaled_i64, scaled_u64],
    });
    regs
}

struct Listed {
    with_costs: String,
    with_facts: String,
}

fn listed(source: &str, ret: Ty) -> Listed {
    let interner = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("parse error"));
    let contexts = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
    let compiled = compile_source_with_externs(&interner, ast, &contexts, regs(), ret);
    let module = &compiled.modules[&compiled.entry_qref];
    Listed {
        with_costs: dump_with_costs(&interner, module, &compiled.laws, &TABLE),
        with_facts: dump_with_facts(&interner, module, &compiled.laws),
    }
}

/// The cost line of each `For`, in the order the listing prints the loops,
/// which puts an outer loop's header before its inner loop's.
fn cost_lines(listing: &str) -> Vec<String> {
    listing
        .lines()
        .filter_map(|line| line.split_once("// cost ").map(|(_, cost)| cost.to_string()))
        .collect()
}

/// `W` of the one `For` whose line is `index`th in the listing.
fn work(listing: &str, index: usize) -> u64 {
    let lines = cost_lines(listing);
    let line = lines
        .get(index)
        .unwrap_or_else(|| panic!("no cost line {index}:\n{listing}"));
    let Some(rest) = line.strip_prefix("W=") else {
        panic!("the loop does not split:\n{listing}")
    };
    let (work, _) = rest.split_once(' ').expect("`W=<ticks> O=..`");
    work.parse().expect("`W` is a number of ticks")
}

#[test]
fn a_branch_weighs_its_lighter_arm() {
    let one_heavy_arm = listed(
        "let s = 0; for i in 0..@n { let y = if i % 2 == 0 { weighed(i) } else { i + 1 }; s = s + y; } s",
        Ty::I64,
    );
    let both_heavy = listed(
        "let s = 0; for i in 0..@n { let y = if i % 2 == 0 { weighed(i) } else { weighed(i + 1) }; s = s + y; } s",
        Ty::I64,
    );
    let light = work(&one_heavy_arm.with_costs, 0);
    assert!(light < WEIGHED, "{}", one_heavy_arm.with_costs);
    assert!(
        work(&both_heavy.with_costs, 0) >= WEIGHED,
        "{}",
        both_heavy.with_costs
    );
}

#[test]
fn an_inner_loop_of_a_constant_trip_weighs_that_many_iterations() {
    let nested = listed(
        "let s = 0; for i in 0..@n { let t = 0; for j in 0..4 { t = t + weighed(j); } s = s + t + i; } s",
        Ty::I64,
    );
    let found = work(&nested.with_costs, 0);
    assert!(
        (4 * WEIGHED..5 * WEIGHED).contains(&found),
        "W={found}:\n{}",
        nested.with_costs
    );
}

#[test]
fn an_inner_loop_of_no_constant_trip_weighs_none_of_its_iterations() {
    let nested = listed(
        "let s = 0; for i in 0..@n { let t = 0; for j in 0..i { t = t + weighed(j); } s = s + t + i; } s",
        Ty::I64,
    );
    let found = work(&nested.with_costs, 0);
    assert!(found < WEIGHED, "W={found}:\n{}", nested.with_costs);
}

#[test]
fn an_extern_that_states_its_cost_weighs_it_in_place_of_its_row() {
    let stated = listed("let s = 0; for i in 0..@n { s = s + weighed(i); } s", Ty::I64);
    let unstated = listed("let s = 0; for i in 0..@n { s = s + unweighed(i); } s", Ty::I64);
    assert_eq!(
        work(&stated.with_costs, 0) - work(&unstated.with_costs, 0),
        WEIGHED - TABLE.extern_call,
        "{}\n{}",
        stated.with_costs,
        unstated.with_costs
    );
}

#[test]
fn each_instance_of_one_signature_weighs_what_it_states() {
    let at_i64 = listed("let s = 0; for i in 0..@n { s = s + scaled(i); } s", Ty::I64);
    let at_u64 = listed(
        "let s = 0u64; for i in 0u64..10u64 { s = s + scaled(i); } s as i64",
        Ty::I64,
    );
    let wide = work(&at_i64.with_costs, 0);
    let narrow = work(&at_u64.with_costs, 0);
    assert!(wide >= WEIGHED, "W={wide}:\n{}", at_i64.with_costs);
    assert!(
        (SCALED_AT_U64..WEIGHED).contains(&narrow),
        "W={narrow}:\n{}",
        at_u64.with_costs
    );
}

#[test]
fn a_heavy_extern_that_states_no_cost_weighs_the_heavy_row() {
    let heavy = listed("let s = 0; for i in 0..@n { s = s + offloaded(i); } s", Ty::I64);
    let found = work(&heavy.with_costs, 0);
    assert!(
        (TABLE.heavy..TABLE.heavy + WEIGHED).contains(&found),
        "W={found}:\n{}",
        heavy.with_costs
    );
}

/// The loop's one stage holds its one cycle, a `Disjoint` store: no stage
/// is free, and that stage runs apart, so the loop is given a cost.
#[test]
fn a_loop_whose_only_stage_is_disjoint_is_given_a_cost() {
    let filled = listed(
        "let c = vec([0, 0, 0]); let z = 7; for i in 0u64..3u64 { c[i] = z; } c[0u64]",
        Ty::I64,
    );
    let facts: Vec<&str> = filled
        .with_costs
        .lines()
        .filter_map(|line| line.split_once("// ").map(|(_, fact)| fact))
        .filter(|fact| !fact.starts_with("cost "))
        .collect();
    assert_eq!(
        facts,
        [
            "L1: cycle Storage(r12) disjoint {index_set}",
            "control upfront"
        ],
        "{}",
        filled.with_costs
    );
    let found = work(&filled.with_costs, 0);
    assert_eq!(found, TABLE.store, "W={found}:\n{}", filled.with_costs);
}

/// Each stage holds an `InOrder` cycle, so none runs apart.
#[test]
fn a_loop_whose_stages_are_all_in_order_runs_in_place() {
    let joined = listed(
        r#"let xs = vec(["a".to_string(), "b".to_string(), "c".to_string()]);
        let s = "".to_string();
        let out = vec([]);
        for x in &xs { s = s + x; out.push(s.to_string()); }
        out.len() as i64"#,
        Ty::I64,
    );
    let stage_facts: Vec<&str> = joined
        .with_costs
        .lines()
        .filter_map(|line| line.split_once("// L").map(|(_, fact)| fact))
        .collect();
    assert!(
        stage_facts.len() == 2 && stage_facts.iter().all(|fact| fact.contains(" in_order ")),
        "{}",
        joined.with_costs
    );
    assert_eq!(
        cost_lines(&joined.with_costs),
        ["in place: no stage runs apart"],
        "{}",
        joined.with_costs
    );
}

#[test]
fn a_float_sum_keeps_its_stages_and_weighs_only_its_free_stage() {
    let summed = listed(
        "let xs = vec([1.5, 2.0, 3.0]); let s = 0.0; for x in &xs { s = s + *x; } s",
        Ty::Float,
    );
    let without_costs: Vec<&str> = summed
        .with_costs
        .lines()
        .filter(|line| !line.contains("// cost "))
        .collect();
    assert_eq!(
        without_costs,
        summed.with_facts.lines().collect::<Vec<_>>(),
        "a cost line is the one line the table adds"
    );
    assert!(
        summed
            .with_costs
            .contains("cycle Carried(r11) in_order law(Op(Add) inexact commutative) {+}"),
        "{}",
        summed.with_costs
    );
    assert_eq!(work(&summed.with_costs, 0), TABLE.load, "{}", summed.with_costs);
}

#[test]
fn a_disjoint_stage_runs_apart_and_weighs_its_work() {
    let per_row = listed(
        "let c = vec([vec([0, 0]), vec([0, 0])]);
        for i in 0u64..2u64 { for j in 0u64..2u64 { c[i][j] = weighed(j as i64); } }
        c[0u64][0u64] + c[1u64][1u64]",
        Ty::I64,
    );
    let one_row = listed(
        "let c = vec([vec([0, 0]), vec([0, 0])]);
        for i in 0u64..2u64 { for j in 0u64..2u64 { c[0u64][j] = weighed(j as i64 + i as i64); } }
        c[0u64][0u64] + c[1u64][1u64]",
        Ty::I64,
    );
    let outer_facts = |listing: &str| -> String {
        listing
            .lines()
            .skip_while(|line| !line.contains(" stages ["))
            .skip(1)
            .take_while(|line| !line.contains("// cost "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert!(
        outer_facts(&per_row.with_costs).contains(" disjoint "),
        "{}",
        per_row.with_costs
    );
    let found = work(&per_row.with_costs, 0);
    assert!(
        (2 * WEIGHED..3 * WEIGHED).contains(&found),
        "W={found}:\n{}",
        per_row.with_costs
    );
    assert!(
        outer_facts(&one_row.with_costs).contains(" in_order "),
        "{}",
        one_row.with_costs
    );
    assert_eq!(
        cost_lines(&one_row.with_costs).first().map(String::as_str),
        Some("in place: W=0"),
        "{}",
        one_row.with_costs
    );
}

/// `n` is the trip count known on entry. A loop whose every exit is its
/// header's runs that many iterations; one that can leave from its body
/// runs at most that many (RFC-0089 rule 5). The two bodies differ only in
/// the exit.
#[test]
fn a_loop_that_can_leave_early_states_its_trip_count_as_a_bound() {
    let upfront = listed(
        "let xs = vec([5, 3, 9]); let at = 0; for i in 0u64..xs.len() { at = at + xs[i]; } at",
        Ty::I64,
    );
    let early = listed(
        "let xs = vec([5, 3, 9]); let at = 0; for i in 0u64..xs.len() { if xs[i] == 9 { at = i as i64; break; }; } at",
        Ty::I64,
    );
    let named = |listing: &str| -> String {
        let lines = cost_lines(listing);
        let [line] = lines.as_slice() else {
            panic!("one cost line:\n{listing}")
        };
        let Some((_, named)) = line.split_once(", n ") else {
            panic!("the loop splits and names its trip count:\n{listing}")
        };
        named.to_string()
    };
    assert!(
        upfront.with_costs.contains("control upfront"),
        "{}",
        upfront.with_costs
    );
    assert!(
        named(&upfront.with_costs).starts_with("= "),
        "{}",
        upfront.with_costs
    );
    assert!(
        early.with_costs.contains("control chained through "),
        "{}",
        early.with_costs
    );
    assert!(
        named(&early.with_costs).starts_with("≤ "),
        "{}",
        early.with_costs
    );
}
