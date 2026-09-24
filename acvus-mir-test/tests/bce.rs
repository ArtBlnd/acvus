//! Which indices `optimize::bce` marks `Proven` (RFC-0047 rule 7), and the
//! validator that refuses a mark the interval domain does not derive. The
//! language indexes a container and never a slice value, so each program
//! reads its bound off the container's `len`.
//!
//! Each program also stands in `acvus-interpreter-test/tests/soundness/bce/`,
//! where it runs at both levels to its pinned value or trap.

use acvus_mir::graph::QualifiedRef;
use acvus_mir::graph::optimize::{Opt, optimize};
use acvus_mir::ir::{IndexBound, InstKind, MirModule};
use acvus_mir::laws::LawTable;
use acvus_mir::printer::dump_with;
use acvus_mir::validate::bounds::check_bounds;
use acvus_mir::validate::type_check::ValidationErrorKind;
use acvus_mir_test::lowered_script;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

struct Optimized {
    module: MirModule,
    laws: LawTable,
    listing: String,
}

impl Optimized {
    fn of(source: &str) -> Self {
        let interner = Interner::new();
        let lowered = lowered_script(&interner, source, &[], vec![])
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        let test = QualifiedRef::root(interner.intern("test"));
        let modules: FxHashMap<QualifiedRef, MirModule> =
            std::iter::once((test, lowered.module)).collect();
        let mut result = optimize(&interner, &lowered.laws, modules, Opt::Full);
        let refused: Vec<String> = result
            .errors
            .iter()
            .flat_map(|(_, errors)| errors.iter().map(|e| e.display(&interner).to_string()))
            .collect();
        assert!(refused.is_empty(), "{source}\n{}", refused.join("\n"));
        let Some(module) = result.modules.remove(&test) else {
            panic!("no module for the script");
        };
        let listing = dump_with(&interner, &module);
        Self {
            module,
            laws: lowered.laws,
            listing,
        }
    }

    fn bounds(&self) -> Vec<IndexBound> {
        self.module
            .main
            .insts
            .iter()
            .filter_map(|inst| match &inst.kind {
                InstKind::Index { bound, .. } | InstKind::IndexSet { bound, .. } => Some(*bound),
                _ => None,
            })
            .collect()
    }

    fn assert_every_index(&self, expected: IndexBound, source: &str) {
        let bounds = self.bounds();
        assert!(!bounds.is_empty(), "the program indexes:\n{source}\n{}", self.listing);
        assert!(
            bounds.iter().all(|b| *b == expected),
            "every index is {expected:?}, found {bounds:?}:\n{source}\n{}",
            self.listing
        );
    }
}

fn proven(source: &str) {
    Optimized::of(source).assert_every_index(IndexBound::Proven, source);
}

fn checked(source: &str) {
    Optimized::of(source).assert_every_index(IndexBound::Checked, source);
}

#[test]
fn a_range_for_up_to_the_container_length_indexes_unchecked() {
    proven("let v = vec([1, 2, 3]); let t = 0; for i in 0..v.len() { t = t + v[i]; } t");
}

#[test]
fn a_while_below_the_container_length_indexes_unchecked() {
    proven(
        "let v = vec([1, 2, 3]); let t = 0; let i = 0; \
         while i < v.len() { t = t + v[i]; i = i + 1; } t",
    );
}

#[test]
fn a_guard_below_the_length_indexes_unchecked() {
    proven("let v = vec([1, 2, 3]); let n = v.len(); let i = n - n + 2; if i < n { v[i] } else { 0 }");
}

#[test]
fn an_element_write_below_the_length_is_unchecked() {
    proven("let v = vec([1, 2, 3]); for i in 0..v.len() { v[i] = 7; } v.len()");
}

#[test]
fn one_past_the_counter_stays_checked() {
    checked("let v = vec([1, 2, 3]); let t = 0; for i in 0..v.len() { t = t + v[i + 1]; } t");
}

/// `i + 1 > i` for the program's trapping `+`, and `n - 1 < n` for its `-`
/// (RFC-0037 rule 3): `v.len() - 1` is one below the length, since a run in
/// which it leaves `u64` ended there, so `i + 1` for `i` below it is below
/// the length. Were the two to wrap, `v.len() - 1` at an empty `v` would be
/// `u64::MAX`, and neither bound would hold.
#[test]
fn one_past_a_counter_below_the_length_less_one_indexes_unchecked() {
    proven(
        "let v = vec([1, 2, 3]); let t = 0; \
         for i in 0..v.len() - 1 { t = t + v[i + 1]; } t",
    );
}

/// `i - 1 < i` for a counter whose start the domain does not know: the
/// counter is below the length, and the program's `-` does not wrap, so
/// the upper bound moves down with it whatever the lower one is.
#[test]
fn one_before_a_counter_from_an_unknown_start_indexes_unchecked() {
    proven(
        "let v = vec([1, 2, 3]); let a = v.len() - v.len() + 1; let t = 0; \
         for i in a..v.len() { t = t + v[i - 1]; } t",
    );
}

#[test]
fn an_index_derived_elsewhere_stays_checked() {
    checked("let v = vec([1, 2, 3]); let n = v.len(); let j = n - n + 5; v[j]");
}

#[test]
fn a_container_written_between_its_length_and_the_index_stays_checked() {
    checked(
        "let v = vec([1, 2, 3]); let n = v.len(); v.push(4); let t = 0; \
         for i in 0..n { t = t + v[i]; } t",
    );
}

#[test]
fn at_most_the_length_stays_checked() {
    checked("let v = vec([1, 2, 3]); let n = v.len(); let i = n - n + 2; if i <= n { v[i] } else { 0 }");
}

#[test]
fn a_proven_mark_the_domain_does_not_derive_is_refused() {
    let source = "let v = vec([1, 2, 3]); let n = v.len(); let j = n - n + 5; v[j]";
    let mut optimized = Optimized::of(source);
    assert_eq!(check_bounds(&optimized.module, &optimized.laws).len(), 0);
    let mut marked = 0;
    for inst in &mut optimized.module.main.insts {
        if let InstKind::Index { bound, .. } = &mut inst.kind {
            *bound = IndexBound::Proven;
            marked += 1;
        }
    }
    assert_eq!(marked, 1, "{}", optimized.listing);
    let errors = check_bounds(&optimized.module, &optimized.laws);
    let kinds: Vec<&ValidationErrorKind> = errors.iter().map(|e| &e.kind).collect();
    assert!(
        matches!(kinds[..], [ValidationErrorKind::UnprovenBound]),
        "one refusal of the unproven mark, found {kinds:?}"
    );
}

#[test]
fn a_proven_mark_the_pass_wrote_is_derived_again() {
    let optimized =
        Optimized::of("let v = vec([1, 2, 3]); let t = 0; for i in 0..v.len() { t = t + v[i]; } t");
    assert!(optimized.bounds().contains(&IndexBound::Proven), "{}", optimized.listing);
    assert_eq!(check_bounds(&optimized.module, &optimized.laws).len(), 0);
}
