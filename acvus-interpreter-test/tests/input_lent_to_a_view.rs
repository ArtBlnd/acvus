//! A `$` input holds a value, and a view is never one (RFC-0062 rule 1): an
//! input nothing types but `&$input` meeting a view parameter is the type
//! that lends the view, and the argument reaches the parameter as that view.

use acvus_interpreter_test::{Helper, Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{ParamTerm, Poly, PolyBuilder, PolyParam, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Compile at `opt` and prepare every module, without running. The run is
/// left out on purpose: a body reading an owned `String` argument crashes
/// the machine whatever typed it, `String` declared included.
fn prepared(i: &Interner, helper: &Helper<'_>, main: &str, opt: Opt) -> Result<(), Refusal> {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        std::slice::from_ref(helper),
        &FxHashMap::default(),
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        Ty::I64,
        opt,
        |_| {},
    )?;
    execute_compiled(
        i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    Ok(())
}

/// One parameter whose type the body's own reads decide, as a host declares
/// an input it has not typed.
fn untyped(i: &Interner, name: &str) -> Vec<PolyParam> {
    let ty = PolyBuilder::new().fresh_ty_var();
    vec![ParamTerm::<Poly>::new(i.intern(name), ty)]
}

/// The template `% let on = string::contains(&$input, "X")`, with the caller
/// handing `$input` in at run time.
fn has_x(i: &Interner) -> Helper<'_> {
    Helper {
        name: "has_x",
        source: "let on = string::contains(&$input, \"X\");\nif on { 1 } else { 0 }\n",
        params: untyped(i, "input"),
    }
}

fn has_one(i: &Interner) -> Helper<'_> {
    Helper {
        name: "has_one",
        source: "if slice::contains(&$xs, &1) { 1 } else { 0 }\n",
        params: untyped(i, "xs"),
    }
}

#[test]
fn an_input_lent_to_a_str_parameter_is_a_string_and_prepares() {
    let i = Interner::new();
    for opt in [Opt::Full, Opt::None] {
        prepared(&i, &has_x(&i), "has_x(\"aXb\".to_string())\n", opt)
            .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")));
    }
}

/// An array and a `Vec` both lend `[T]`, so a lend at a slice parameter
/// alone does not decide which one the input holds.
#[test]
fn an_input_lent_only_to_a_slice_parameter_is_undecided() {
    let i = Interner::new();
    for opt in [Opt::Full, Opt::None] {
        let refused = match prepared(&i, &has_one(&i), "1\n", opt) {
            Ok(()) => panic!("{opt:?} admitted an input typed only by a slice parameter"),
            Err(r) => r.messages.join("\n"),
        };
        assert!(
            refused.contains("[has_one] cannot infer type: resolved to &_"),
            "{opt:?}: {refused}"
        );
    }
}
