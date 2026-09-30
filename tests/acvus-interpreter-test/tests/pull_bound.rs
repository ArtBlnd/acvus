//! RFC-0089 rules 1 and 5 at the listing: a pull loop inside a search's
//! predicate runs ahead of the search's exit only where the call that made
//! its iterator bounds its pulls (RFC-0082 rule 4's `len(ret)`), and
//! nothing but the loop's header touches the iterator; and a pull runs
//! ahead of its own loop's body exit only where its instance states
//! `returns`.

use acvus_extern::{ExternType, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::check_source;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

pub struct CountdownBody {
    left: i64,
}

/// `n` ones from a nonnegative `n`, and no end from a negative one.
#[derive(ExternType)]
#[extern_type(name = "Countdown")]
#[repr(transparent)]
pub struct Countdown(CountdownBody);

#[extern_fn(effect = pure, total)]
fn countdown(n: i64) -> Countdown {
    Countdown(CountdownBody { left: n })
}

#[extern_fn(instance_of = acvus_ext::iter_sig::next, effect = pure, total)]
fn next_countdown(it: &mut Countdown) -> Option<i64> {
    let left = it.0.left;
    if left == 0 {
        return None;
    }
    if left > 0 {
        it.0.left = left - 1;
    }
    Some(1)
}

/// A countdown whose `next` states nothing of how it ends.
#[derive(ExternType)]
#[extern_type(name = "Unstated")]
#[repr(transparent)]
pub struct Unstated(CountdownBody);

#[extern_fn(effect = pure, total)]
fn unstated(n: i64) -> Unstated {
    Unstated(CountdownBody { left: n })
}

#[extern_fn(instance_of = acvus_ext::iter_sig::next, effect = pure)]
fn next_unstated(it: &mut Unstated) -> Option<i64> {
    let left = it.0.left;
    if left <= 0 {
        return None;
    }
    it.0.left = left - 1;
    Some(left)
}

fn countdown_registry<Rt>() -> Registry<Rt>
where
    Rt: Runtime,
{
    extern_registry! {
        ns: "cd",
        types: [Countdown, Unstated],
        fns: [countdown, next_countdown, unstated, next_unstated],
    }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(countdown_registry());
    registries
}

fn listing(source: &str, ret: Ty) -> String {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("the script parses"));
    let compiled = check_source(&i, ast, &FxHashMap::default(), registries(), ret, Opt::Full, |_| {})
        .unwrap_or_else(|refusal| panic!("{}", refusal.messages.join("; ")));
    acvus_mir::printer::dump_with_facts(&i, &compiled.modules[&compiled.entry_qref], &compiled.laws)
}

fn outer_free_stage(source: &str) -> String {
    let listing = listing(source, Ty::U64);
    let mut lines = listing.lines().skip_while(|line| !line.contains("for range("));
    lines.next();
    lines
        .next()
        .filter(|line| line.contains(": free {"))
        .unwrap_or_else(|| panic!("the outer `for` has a free stage:\n{listing}"))
        .to_string()
}

/// The test of the pulled `Option` in a pull loop's header, which no other
/// operation of these programs writes.
const INNER_PULL_TEST: &str = "test_variant";

const CHARS_SEARCH: &str = "let xs = vec([\"12345\".to_string(), \"99999\".to_string()]); let at = 99u64;
    for i in 0u64..xs.len() {
        let s = 0u32; let cs = xs[i].chars();
        while let Some(c) = cs.next() { s = s + c.to_digit(10u32).unwrap(); }
        if s > 20u32 { at = i; break; };
    }
    at";

/// Corpus row S07: `chars` states `len(ret) <= len(s)`, so the digit sum
/// finishes on every run and is the search's free work.
#[test]
fn a_pull_over_chars_in_a_search_predicate_runs_ahead_of_the_exit() {
    let free = outer_free_stage(CHARS_SEARCH);
    assert!(free.contains("call chars") && free.contains(INNER_PULL_TEST), "{free}");
}

/// A pull of an iterator whose making call states no bound may not end, so
/// it waits for the exit: here the second element's run never ends, and
/// the program leaves at the first.
#[test]
fn a_pull_whose_making_call_states_no_bound_is_held_back() {
    let free = outer_free_stage(
        "let xs = vec([3, -1]); let at = 99u64;
         for i in 0u64..xs.len() {
             let s = 0; let it = cd::countdown(xs[i]);
             while let Some(x) = it.next() { s = s + x; }
             if s > 2 { at = i; break; };
         }
         at",
    );
    assert!(!free.contains(INNER_PULL_TEST), "{free}");
}

/// An iterator pulled once more before its loop is touched past the
/// header: the count bound no longer names what the loop reads.
#[test]
fn a_bounded_iterator_touched_outside_its_header_is_held_back() {
    let free = outer_free_stage(
        "let xs = vec([\"12345\".to_string(), \"99999\".to_string()]); let at = 99u64;
         for i in 0u64..xs.len() {
             let s = 0u32; let cs = xs[i].chars(); cs.next();
             while let Some(c) = cs.next() { s = s + c.to_digit(10u32).unwrap(); }
             if s > 20u32 { at = i; break; };
         }
         at",
    );
    assert!(!free.contains(INNER_PULL_TEST), "{free}");
}

/// A pull whose instance states no `returns` may not finish, so it may not
/// run ahead of the body's `break`: the loop is no pull loop.
#[test]
fn a_pull_whose_instance_states_no_returns_does_not_run_ahead_of_a_body_exit() {
    let source = "let it = cd::unstated(5); let found = 0;
        while let Some(x) = it.next() { if x < 3 { found = x; break; }; }
        found";
    let listed = listing(source, Ty::I64);
    assert!(!listed.contains(" stages ["), "{listed}");
}

/// The same loop over an iterator whose `next` states `total` is a pull
/// loop running its pulls ahead of the `break`.
#[test]
fn a_pull_whose_instance_states_total_runs_ahead_of_a_body_exit() {
    let source = "let it = cd::countdown(5); let found = 0;
        while let Some(x) = it.next() { if x > 0 { found = x; break; }; }
        found";
    let listed = listing(source, Ty::I64);
    assert!(listed.contains("while ") && listed.contains(" stages ["), "{listed}");
}
