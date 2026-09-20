//! A local function's parameters are bound by its declaration: a `$name` in
//! the body names the declared parameter of that name, the declared order is
//! the order the call passes arguments in, and a `$name` the declaration does
//! not name is refused.

use acvus_interpreter::Value;
use acvus_interpreter_test::{Helper, Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{LenTerm, Mutability, ParamTerm, Poly, PolyParam, Ty, TypeArg, lift_to_poly};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

const READS_STOP_THEN_XS: &str = "if $stop { $xs[0] } else { $xs[1] }\n";

fn sig(i: &Interner, params: &[(&str, Ty)]) -> Vec<PolyParam> {
    params
        .iter()
        .map(|(name, ty)| ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(ty)))
        .collect()
}

fn lent_numbers() -> Ty {
    let numbers = Ty::Array(Box::new(Ty::I64), LenTerm::Known(3));
    Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(numbers)))
}

fn compile_and_run(
    i: &Interner,
    helper: Helper<'_>,
    main: &str,
    opt: Opt,
) -> Result<Value, Refusal> {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        std::slice::from_ref(&helper),
        &FxHashMap::default(),
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        Ty::I64,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()))
}

/// The integer a program yields at both optimization levels. Disagreement is
/// the differential's own contract, so it fails here rather than being
/// reported as one number.
fn integer_at_both_levels(i: &Interner, helper: fn(&Interner) -> Helper<'_>, main: &str) -> i64 {
    let full = compile_and_run(i, helper(i), main, Opt::Full)
        .unwrap_or_else(|r| panic!("Opt::Full refused:\n  {}", r.messages.join("\n  ")))
        .as_int();
    let none = compile_and_run(i, helper(i), main, Opt::None)
        .unwrap_or_else(|r| panic!("Opt::None refused:\n  {}", r.messages.join("\n  ")))
        .as_int();
    assert_eq!(full, none, "the two optimization levels read one element");
    full
}

fn refusal(i: &Interner, helper: Helper<'_>, main: &str) -> String {
    match compile_and_run(i, helper, main, Opt::Full) {
        Ok(value) => panic!("expected a refusal, ran to {value:?}"),
        Err(r) => r.messages.join("\n"),
    }
}

fn lends_then_stops(i: &Interner) -> Helper<'_> {
    Helper {
        name: "pick",
        source: READS_STOP_THEN_XS,
        params: sig(i, &[("xs", lent_numbers()), ("stop", Ty::Bool)]),
    }
}

fn stops_then_lends(i: &Interner) -> Helper<'_> {
    Helper {
        name: "pick",
        source: READS_STOP_THEN_XS,
        params: sig(i, &[("stop", Ty::Bool), ("xs", lent_numbers())]),
    }
}

fn lends_then_stops_reading_only_xs(i: &Interner) -> Helper<'_> {
    Helper {
        name: "pick",
        source: "$xs[0]\n",
        params: sig(i, &[("xs", lent_numbers()), ("stop", Ty::Bool)]),
    }
}

// -- The declaration binds, in either order ---------------------------

#[test]
fn a_parameter_declared_before_the_one_read_first_keeps_its_type_and_position() {
    let i = Interner::new();
    let taken = integer_at_both_levels(
        &i,
        lends_then_stops,
        "let v = [10, 20, 30];\npick(&v, true)\n",
    );
    assert_eq!(taken, 10);
    let skipped = integer_at_both_levels(
        &i,
        lends_then_stops,
        "let v = [10, 20, 30];\npick(&v, false)\n",
    );
    assert_eq!(skipped, 20);
}

#[test]
fn a_parameter_declared_in_the_order_the_body_reads_keeps_its_type_and_position() {
    let i = Interner::new();
    let taken = integer_at_both_levels(
        &i,
        stops_then_lends,
        "let v = [10, 20, 30];\npick(true, &v)\n",
    );
    assert_eq!(taken, 10);
    let skipped = integer_at_both_levels(
        &i,
        stops_then_lends,
        "let v = [10, 20, 30];\npick(false, &v)\n",
    );
    assert_eq!(skipped, 20);
}

// -- What the declaration does not name --------------------------------

#[test]
fn a_dollar_name_the_declaration_does_not_name_is_refused_naming_it() {
    let i = Interner::new();
    let helper = Helper {
        name: "pick",
        source: READS_STOP_THEN_XS,
        params: sig(&i, &[("xs", lent_numbers())]),
    };
    let refused = refusal(&i, helper, "let v = [10, 20, 30];\npick(&v)\n");
    assert!(refused.contains("undefined variable `$stop`"), "{refused}");
    assert!(
        !refused.contains("arguments"),
        "the call passes the one argument the declaration names: {refused}"
    );
}

#[test]
fn a_declared_parameter_the_body_never_reads_is_still_taken_by_the_call() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        lends_then_stops_reading_only_xs,
        "let v = [10, 20, 30];\npick(&v, true)\n",
    );
    assert_eq!(value, 10);
}
