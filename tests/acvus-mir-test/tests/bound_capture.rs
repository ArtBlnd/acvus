//! A binding of `$n` fills the input `$n` and nothing else: a local, a
//! closure's parameter or a capture named `n` keeps the value the program
//! gives it (RFC-0071 rule 4, RFC-0054 rule 6).

use acvus_mir::graph::BoundValue;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{ParamTerm, Poly, PolyParam, TyTerm};
use acvus_mir_test::compile_script_bound;
use acvus_utils::Interner;

struct Probe {
    variant: &'static str,
    source: &'static str,
    returns_with_n_100: i64,
    unoptimized_main_makes: &'static str,
}

const PROBES: &[Probe] = &[
    Probe {
        variant: "a capture of a local named like the input",
        source: "let n = 5; let f = |x| -> x + n; f(0)",
        returns_with_n_100: 5,
        unoptimized_main_makes: "closure L0 [5 (",
    },
    Probe {
        variant: "a capture of the local and of the input in one closure",
        source: "let n = 5; let f = |x| -> x + n + $n; f(0)",
        returns_with_n_100: 105,
        unoptimized_main_makes: "closure L0 [5 (r1), 100 (r2)]",
    },
    Probe {
        variant: "a closure parameter named like the input",
        source: "let f = |n| -> n + 1; f(5)",
        returns_with_n_100: 6,
        unoptimized_main_makes: "closure L0 []",
    },
    Probe {
        variant: "the input read beside a closure parameter of its name",
        source: "let f = |n| -> n + $n; f(5)",
        returns_with_n_100: 105,
        unoptimized_main_makes: "closure L0 [100 (",
    },
    Probe {
        variant: "a nested closure capturing the local",
        source: "let n = 5; let f = |x| -> { let g = |y| -> y + n; g(x) }; f(0)",
        returns_with_n_100: 5,
        unoptimized_main_makes: "closure L0 [5 (",
    },
    Probe {
        variant: "a nested closure capturing the local and the input",
        source: "let n = 5; let f = |x| -> { let g = |y| -> y + n + $n; g(x) }; f(0)",
        returns_with_n_100: 105,
        unoptimized_main_makes: "closure L0 [5 (r1), 100 (r2)]",
    },
    Probe {
        variant: "a shadowing `let n` in the closure, read as the local",
        source: "let f = |x| -> { let n = 7; n + x }; f(0)",
        returns_with_n_100: 7,
        unoptimized_main_makes: "closure L0 []",
    },
    Probe {
        variant: "a shadowing `let n` in the closure beside a read of the input",
        source: "let f = |x| -> { let n = 7; $n + x }; f(0)",
        returns_with_n_100: 100,
        unoptimized_main_makes: "closure L0 [100 (",
    },
];

fn n_bound_to_100() -> Vec<(&'static str, BoundValue)> {
    vec![("n", BoundValue::Int(100))]
}

fn main_listing(ir: &str) -> &str {
    ir.split("=== closure")
        .next()
        .expect("a listing has a main")
}

fn compiled_ir(source: &str, opt: Opt) -> String {
    let interner = Interner::new();
    compile_script_bound(&interner, source, &[], &n_bound_to_100(), opt)
        .unwrap_or_else(|e| panic!("`{source}` compiles: {e}"))
        .ir
}

#[test]
fn a_binding_leaves_every_local_of_its_name_to_the_program() {
    for probe in PROBES {
        let ir = compiled_ir(probe.source, Opt::Full);
        let returns = probe.returns_with_n_100;
        assert!(
            main_listing(&ir).contains(&format!("return {returns} (")),
            "{}: `{}` with n=100 returns {returns}:\n{ir}",
            probe.variant,
            probe.source
        );
    }
}

#[test]
fn a_closure_takes_the_local_or_the_input_its_body_names() {
    for probe in PROBES {
        let ir = compiled_ir(probe.source, Opt::None);
        assert!(
            main_listing(&ir).contains(probe.unoptimized_main_makes),
            "{}: `{}` with n=100 makes its closure with `{}`:\n{ir}",
            probe.variant,
            probe.source,
            probe.unoptimized_main_makes
        );
    }
}

fn helper_f_declaring_n(
    interner: &Interner,
    body: &'static str,
) -> Vec<(&'static str, &'static str, Vec<PolyParam>)> {
    vec![(
        "f",
        body,
        vec![ParamTerm::<Poly>::new(interner.intern("n"), TyTerm::I64)],
    )]
}

fn compiled_with_f(body: &'static str, opt: Opt) -> acvus_mir_test::BoundModule {
    let interner = Interner::new();
    let helpers = helper_f_declaring_n(&interner, body);
    let module = compile_script_bound(&interner, "f(5) + $n", &helpers, &n_bound_to_100(), opt)
        .unwrap_or_else(|e| panic!("`f` as `{body}` compiles: {e}"));
    assert!(
        module.inputs.is_empty(),
        "the binding fixes the one input: {:?}",
        module.inputs
    );
    module
}

#[test]
fn a_declared_parameter_named_like_a_binding_is_filled_by_its_call() {
    let module = compiled_with_f("$n + 1", Opt::Full);
    assert!(
        main_listing(&module.ir).contains("return 106 ("),
        "`f(5) + $n`, `f` as `$n + 1`, with n=100 returns 106:\n{}",
        module.ir
    );
}

#[test]
fn a_closure_in_a_function_captures_its_declared_parameter_and_not_the_binding() {
    for opt in [Opt::Full, Opt::None] {
        let module = compiled_with_f("let g = |x| -> x + $n + 1; g(0)", opt);
        let f = &module.helper_ir_by_name["f"];
        let f_main = main_listing(f);
        assert!(
            !f_main.contains("100"),
            "the binding n=100 reaches nothing in `f`:\n{f}"
        );
        if opt == Opt::None {
            assert!(
                f_main.contains("closure L0 [r"),
                "`f` makes its closure from its parameter `n`:\n{f}"
            );
        }
    }
}
