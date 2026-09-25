//! `anyorder` is lexical (RFC-0007 rule 2): a closure defined inside the block
//! belongs to it, so each effectful call its body makes takes the closure's
//! entry `Order` and the body yields their merge, while a closure defined
//! outside the block keeps its calls in order wherever it is called.

use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::check_graph;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

const DEFINED_INSIDE: &str = include_str!("../par_corpus/f10_anyorder_helper_print.acvus");

const DEFINED_OUTSIDE: &str = "let n = 0u64;
let each = |v, f| -> {
    for x in &v {
        f(*x);
    }
    v.len()
};
anyorder {
    n = each(vec([5, 3]), |x| -> io::print(&x.to_string()));
}
n";

fn registries() -> Vec<acvus_extern::Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(acvus_ext::io_registry::<AcvusRuntime>());
    registries
}

/// What `acvus mir` prints of the module, the facts of each `For` included.
fn listing(source: &str, ret: Ty, opt: Opt) -> String {
    let i = Interner::new();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("the source parses"));
    let compiled = check_graph(
        &i,
        parsed,
        &[],
        &FxHashMap::default(),
        registries(),
        ret,
        opt,
        |_| {},
    )
    .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")));
    acvus_mir::printer::dump_with_facts(&i, &compiled.modules[&compiled.entry_qref], &compiled.laws)
}

fn order_cycles(listing: &str) -> Vec<&str> {
    listing
        .lines()
        .filter(|line| line.contains("cycle Order("))
        .collect()
}

#[test]
fn a_closure_defined_inside_takes_the_entry_order() {
    for opt in [Opt::None, Opt::Full] {
        let shown = listing(DEFINED_INSIDE, Ty::U64, opt);
        let cycles = order_cycles(&shown);
        assert!(
            !cycles.is_empty(),
            "{opt:?}: the helper's loop threads an Order\n{shown}"
        );
        for cycle in cycles {
            assert!(
                cycle.ends_with("any_order law(Order exact commutative) {merge}"),
                "{opt:?}: {cycle}\n{shown}"
            );
        }
    }
}

#[test]
fn a_closure_defined_outside_keeps_its_order() {
    for opt in [Opt::None, Opt::Full] {
        let shown = listing(DEFINED_OUTSIDE, Ty::U64, opt);
        let cycles = order_cycles(&shown);
        assert!(
            !cycles.is_empty(),
            "{opt:?}: the helper's loop threads an Order\n{shown}"
        );
        for cycle in cycles {
            assert!(cycle.contains(" in_order "), "{opt:?}: {cycle}\n{shown}");
        }
    }
}

fn returns_a_merge(listing: &str, returned: &str) -> bool {
    let line = listing
        .lines()
        .find(|line| line.contains(&format!("return {returned} ")))
        .unwrap_or_else(|| panic!("a `return {returned}` line\n{listing}"));
    let order = line
        .rsplit_once('[')
        .and_then(|(_, rest)| rest.strip_suffix(']'))
        .unwrap_or_else(|| panic!("the return yields an Order: {line}"));
    listing
        .lines()
        .any(|line| line.contains(&format!("{order} = merge ")))
}

#[test]
fn a_return_inside_the_region_yields_the_merge() {
    let in_closure = "let g = 0; anyorder { let f = |x| -> { io::print(\"a\"); \
                      if x > 0 { return 1; }; io::print(\"b\"); 2 }; g = f(1); } g";
    let in_block = "let f = |x| -> { anyorder { io::print(\"a\"); \
                    if x > 0 { return 1; }; io::print(\"b\"); } 2 }; f(1)";
    for source in [in_closure, in_block] {
        let shown = listing(source, Ty::I64, Opt::None);
        assert!(returns_a_merge(&shown, "1"), "{source}\n{shown}");
    }
}
