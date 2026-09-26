//! RFC-0099 at the consumer's contract: every declaration stating a step
//! agrees with its handler up to its bound, a step that lies is caught, and
//! a pipeline outside rule 2's form keeps its handler calls.

use std::ops::Deref;

use acvus_extern::{
    Cross, Ctx, Instance, Later, PassedByValue, Registry, Runtime, Stored, Var, extern_fn,
    extern_registry, kind,
};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::corpus::Outcome;
use acvus_interpreter_test::step_model::{self, Broken, ExternName, Ran, means as means_model};
use acvus_mir::graph::optimize::Opt;
use acvus_utils::Interner;

/// The attempts share the machine with the rest of a workspace test run.
const WORKERS: usize = 8;

/// Each declaration's bound is the largest this many programs reach.
const CASES_PER_DECLARATION: usize = 700;

#[test]
fn every_stepping_declaration_agrees_with_its_handler_up_to_its_bound() {
    let interner = Interner::new();
    let declarations = step_model::stepping_declarations(&interner, acvus_ext::std_registries());
    assert!(
        !declarations.is_empty(),
        "the standard registries state steps (RFC-0099 rule 1)"
    );
    let plans: Vec<step_model::Plan> = declarations
        .iter()
        .map(|declaration| {
            step_model::plan_within(declaration, CASES_PER_DECLARATION)
                .unwrap_or_else(|why| panic!("`{}` has no plan: {why}", declaration.name))
        })
        .collect();
    for plan in &plans {
        eprintln!(
            "{}: {:?}, {} cases\n  {}",
            plan.declaration,
            plan.bound,
            plan.cases.len(),
            plan.positions.join("\n  ")
        );
    }
    let broken = step_model::check(&plans, acvus_ext::std_registries, WORKERS);
    assert!(
        broken.is_empty(),
        "{} cases break the promise; the first ones: {:#?}",
        broken.len(),
        &broken[..broken.len().min(5)]
    );
}

/// States `count`'s step and counts one more element than there are.
#[extern_fn(effect = E, step(state s = 0; s = s +% 1; finish s))]
fn counts_one_too_many<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, acvus_ext::iter_sig::next<I, T, E, Rt>, I, Rt, Later>,
) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut n: i64 = 1;
    while it.call(ctx, ()).is_some() {
        n = n.wrapping_add(1);
    }
    n
}

fn with_a_lying_step() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries();
    registries.push(extern_registry! {
        ns: "lies",
        fns: [counts_one_too_many],
    });
    registries
}

#[test]
fn a_step_its_handler_does_not_keep_is_caught() {
    let interner = Interner::new();
    let lying = step_model::stepping_declarations(&interner, with_a_lying_step())
        .into_iter()
        .find(|declaration| declaration.name == "counts_one_too_many")
        .expect("the registry holds the lying declaration");
    let plan = step_model::plan_within(&lying, CASES_PER_DECLARATION).expect("a plan");
    let broken = step_model::check(&[plan], with_a_lying_step, WORKERS);
    assert!(
        broken.iter().all(|case| matches!(case, Broken::Differ { .. })) && !broken.is_empty(),
        "the fused loop runs the step, the handler counts one more: {broken:#?}"
    );
}

#[test]
fn every_declared_term_holds_against_its_handler() {
    let interner = Interner::new();
    let declarations = means_model::meaning_declarations(&interner, acvus_ext::std_registries());
    let plans: Vec<means_model::Plan> = declarations
        .iter()
        .map(|declaration| {
            means_model::plan(declaration).unwrap_or_else(|why| {
                panic!("`{}` has no plan: {why}", declaration.extern_name.name)
            })
        })
        .collect();
    let mut checked: Vec<&str> = plans.iter().map(|plan| plan.declaration.as_str()).collect();
    checked.dedup();
    for wanted in [
        "new", "to_string", "clone", "or_insert", "get", "get_mut", "insert", "remove",
        "contains_key", "contains", "unwrap_or", "is_some",
    ] {
        assert!(checked.contains(&wanted), "`{wanted}` states no term: {checked:?}");
    }
    for plan in &plans {
        eprintln!(
            "{:?}: {} cases\n  {}",
            plan.cases.first().map(|case| &case.declaration),
            plan.cases.len(),
            plan.positions.join("\n  ")
        );
    }
    let broken = means_model::check(&plans, acvus_ext::std_registries, WORKERS);
    assert!(
        broken.is_empty(),
        "{} cases break the promise; the first ones: {:#?}",
        broken.len(),
        &broken[..broken.len().min(5)]
    );
}

/// `*a + 1`, trapping where the language's `+` does.
#[extern_fn(effect = pure, means(*a + 1))]
fn one_more(a: &i64) -> i64 {
    match a.checked_add(1) {
        Some(sum) => sum,
        None => panic!("the sum leaves i64"),
    }
}

fn with_an_operator_term() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries();
    registries.push(extern_registry! {
        ns: "ops",
        fns: [one_more],
    });
    registries
}

/// A term's operator is the language's, overflow trap included: at
/// `i64::MAX` the handler and the term both trap.
#[test]
fn an_operator_term_its_handler_keeps_holds() {
    let interner = Interner::new();
    let declaration = means_model::meaning_declarations(&interner, with_an_operator_term())
        .into_iter()
        .find(|declaration| declaration.extern_name.name == "one_more")
        .expect("the registry holds the declaration");
    let plan = means_model::plan(&declaration).expect("a plan");
    let broken = means_model::check(&[plan], with_an_operator_term, WORKERS);
    assert!(broken.is_empty(), "{broken:#?}");
}

/// States `clone`'s term `*a` and returns one more.
#[extern_fn(effect = pure, means(*a))]
fn copies_one_more(a: &i64) -> i64 {
    *a + 1
}

fn with_a_lying_term() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries();
    registries.push(extern_registry! {
        ns: "lies",
        fns: [copies_one_more],
    });
    registries
}

#[test]
fn a_term_its_handler_does_not_keep_is_caught() {
    let interner = Interner::new();
    let lying = means_model::meaning_declarations(&interner, with_a_lying_term())
        .into_iter()
        .find(|declaration| declaration.extern_name.name == "copies_one_more")
        .expect("the registry holds the lying declaration");
    let plan = means_model::plan(&lying).expect("a plan");
    let broken = means_model::check(&[plan], with_a_lying_term, WORKERS);
    assert!(
        !broken.is_empty()
            && broken
                .iter()
                .all(|case| matches!(case, means_model::Broken::Differ { .. })),
        "the term reads the argument, the handler adds one: {broken:#?}"
    );
}

/// `insert`'s handler held to `remove`'s term: every case the key is
/// present or the value is stored tells them apart.
#[test]
fn an_entry_term_its_handler_does_not_keep_is_caught() {
    let interner = Interner::new();
    let declarations = means_model::meaning_declarations(&interner, acvus_ext::std_registries());
    let term_of = |namespace: &str, name: &str| {
        declarations
            .iter()
            .find(|declaration| {
                declaration.extern_name.namespace.as_deref() == Some(namespace)
                    && declaration.extern_name.name == name
            })
            .unwrap_or_else(|| panic!("`{namespace}::{name}` states a term"))
    };
    let removing = term_of("map", "remove").means().clone();
    let lying = term_of("map", "insert").clone().held_to(removing);
    let plan = means_model::plan(&lying).expect("a plan");
    let broken = means_model::check(&[plan], acvus_ext::std_registries, WORKERS);
    assert!(
        !broken.is_empty()
            && broken
                .iter()
                .all(|case| matches!(case, means_model::Broken::Differ { .. })),
        "insert stores the value, the term takes the entry out: {broken:#?}"
    );
}

fn iter_name(name: &str) -> ExternName {
    ExternName {
        namespace: Some("iter".to_string()),
        name: name.to_string(),
    }
}

fn at_both_levels(source: &str) -> Ran {
    let full = step_model::run_at(source, Opt::Full, acvus_ext::std_registries);
    let none = step_model::run_at(source, Opt::None, acvus_ext::std_registries);
    assert!(
        matches!(full.outcome, Outcome::Value(_)),
        "the program runs: {:?}",
        full.outcome
    );
    assert_eq!(full.outcome, none.outcome, "one program at both levels");
    full
}

fn keeps(full: &Ran, calls: &[&str]) {
    for name in calls {
        assert!(
            full.calls.contains(&iter_name(name)),
            "`{name}` is left to its handler, and the fused code does not call it: {:?}",
            full.calls
        );
    }
}

fn fuses(full: &Ran, calls: &[&str]) {
    for name in calls {
        assert!(
            !full.calls.contains(&iter_name(name)),
            "`{name}` is fused, and the code still calls it: {:?}",
            full.calls
        );
    }
}

#[test]
fn a_pipeline_consumed_where_it_is_built_is_fused() {
    let full = at_both_levels(
        "let v = vec([1, 2, 3]);\nv.into_iter().filter(|x| -> *x > 1).map(|x| -> x * 2).sum()\n",
    );
    fuses(&full, &["filter", "map", "sum"]);
}

#[test]
fn a_pipeline_value_read_on_two_paths_is_not_fused() {
    let full = at_both_levels(
        "let v = vec([1, 2, 3]);\nlet c = v.len() > 2;\nlet m = v.into_iter().map(|x| -> x + 1);\n\
         if c { m.sum() } else { m.count() }\n",
    );
    keeps(&full, &["map", "sum", "count"]);
}

#[test]
fn a_pipeline_passed_to_a_call_stating_no_step_is_not_fused() {
    let full = at_both_levels(
        "let v = vec([1, 2, 3]);\nv.into_iter().map(|x| -> x + 1).take(2).sum()\n",
    );
    keeps(&full, &["map", "take"]);
}

#[test]
fn a_pipeline_returned_from_its_body_is_not_fused_there() {
    let full = at_both_levels(
        "let v = vec([1, 2, 3]);\nlet build = |w| -> w.into_iter().map(|x| -> x + 1);\n\
         build(v).count()\n",
    );
    keeps(&full, &["map"]);
}

/// A closure moves, so its value has a second reader only on another path.
#[test]
fn a_closure_argument_with_a_second_reader_is_not_fused() {
    let full = at_both_levels(
        "let v = vec([1, 2, 3]);\nlet c = v.len() > 2;\nlet f = |x| -> x + 1;\n\
         if c { v.into_iter().map(f).sum() } else { v.into_iter().map(f).count() }\n",
    );
    keeps(&full, &["map", "sum", "count"]);
}

#[test]
fn a_stepping_source_that_is_a_parameter_is_not_fused() {
    let full = at_both_levels(
        "let v = vec([1, 2, 3]);\nlet total = |m| -> m.sum();\ntotal(v.into_iter().map(|x| -> x + 1))\n",
    );
    keeps(&full, &["map", "sum"]);
}

#[test]
fn a_pipeline_built_and_consumed_in_one_closure_body_is_fused_there() {
    let full = at_both_levels(
        "let v = vec([1, 2, 3]);\nlet total = |w| -> w.into_iter().map(|x| -> x * 2).sum();\ntotal(v)\n",
    );
    fuses(&full, &["map", "sum"]);
}

/// `optimize::forward` removed a `While`'s parameterless body block once the
/// body folded to a jump, and the loop named a block no longer there; a
/// filter whose predicate is constantly false folds its fused body so.
#[test]
fn a_pull_loop_whose_body_folds_away_keeps_its_body_block() {
    at_both_levels(
        "let it = vec([0.0]).into_iter();\nlet s = vec([]);\n\
         while let Some(x) = it.next() { if false { s.push(x); }; }\n\
         let t = 0.0;\nfor y in &s { t = t + *y; }\nt\n",
    );
    at_both_levels("let xs = vec([0.5]).into_iter().filter(|x| -> false).collect();\nxs.into_iter().sum()\n");
}

/// A capturing closure the fused loop calls once per element, whose own body
/// holds a fused loop reading the capture.
#[test]
fn a_capturing_closure_is_called_once_per_element_through_its_storage() {
    let full = at_both_levels(
        "let values = vec([vec([1, 2]), vec([3, 4])]);\nlet half = 5;\n\
         values.as_iter().map(|row| -> row.as_iter().map(|x| -> half + *x).sum()).sum()\n",
    );
    fuses(&full, &["map", "sum"]);
}
