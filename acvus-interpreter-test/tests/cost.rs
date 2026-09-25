//! Each clause of RFC-0066 rule 8 at the listing `acvus mir` prints: the
//! cost line under a `For`, computed from `analysis::loop_deps`' free
//! stages and a table. The table here gives each family a weight no sum of
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

#[test]
fn a_loop_with_no_free_stage_runs_in_place() {
    let joined = listed(
        r#"let xs = vec(["a".to_string(), "b".to_string(), "c".to_string()]);
        let s = "".to_string();
        let out = vec([]);
        for x in &xs { s = s + x; out.push(s.to_string()); }
        out.len() as i64"#,
        Ty::I64,
    );
    assert_eq!(
        cost_lines(&joined.with_costs),
        ["in place: no free stage"],
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
            .contains("cycle Carried(r8) in_order law(Op(Add) inexact commutative) {+}"),
        "{}",
        summed.with_costs
    );
    assert_eq!(work(&summed.with_costs, 0), TABLE.load, "{}", summed.with_costs);
}
