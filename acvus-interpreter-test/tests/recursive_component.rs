//! RFC-0042 rule 5 and RFC-0025 rule 5: the functions of one call-graph
//! component are typed together, and each member's effect is closed over
//! what it reaches through the cycle.

use std::collections::BTreeSet;
use std::sync::Arc;

use acvus_interpreter::Value;
use acvus_interpreter_test::{
    Helper, Refusal, check_graph, execute_compiled, int_context, split_context,
};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    Access, Bindings, CompilationGraph, Context, FnKind, Function, Inputs, ParsedAst, QualifiedRef,
    extract, infer,
};
use acvus_mir::ty::{
    Effect, Flows, ParamTerm, Poly, PolyBuilder, PolyParam, Ty, TyTerm, lift_declaration,
    lift_to_poly,
};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

fn param(i: &Interner, name: &str, ty: Ty) -> PolyParam {
    ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(&ty))
}

fn helper<'a>(i: &Interner, name: &'a str, source: &'a str) -> Helper<'a> {
    Helper {
        name,
        source,
        params: vec![param(i, "n", Ty::I64)],
    }
}

#[derive(Debug, PartialEq)]
struct Committed {
    context: String,
    value: i64,
}

struct Ran {
    value: Value,
    committed: Vec<Committed>,
}

fn compile_and_run(
    i: &Interner,
    helpers: &[Helper<'_>],
    main: &str,
    ret: Ty,
    opt: Opt,
) -> Result<Ran, Refusal> {
    let (context_types, snapshot) = split_context(i, int_context(i, "c", 5));
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        helpers,
        &context_types,
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        ret,
        opt,
        |_| {},
    )?;
    let (_, mut interp) =
        execute_compiled(i, cr, snapshot, Arc::new(acvus_interpreter::SequentialExecutor));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    let value = runtime
        .block_on(interp.execute())
        .expect("the seeds hold every context the run fetches");
    let committed = interp
        .take_writes()
        .into_iter()
        .map(|write| Committed {
            context: write.key,
            value: write.value.as_int(),
        })
        .collect();
    Ok(Ran { value, committed })
}

fn ran_at_both_levels(i: &Interner, helpers: &[Helper<'_>], main: &str, ret: Ty) -> Vec<Ran> {
    [Opt::None, Opt::Full]
        .into_iter()
        .map(|opt| {
            compile_and_run(i, helpers, main, ret.clone(), opt)
                .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")))
        })
        .collect()
}

fn integer_at_both_levels(i: &Interner, helpers: &[Helper<'_>], main: &str) -> i64 {
    let ran = ran_at_both_levels(i, helpers, main, Ty::I64);
    let values: Vec<i64> = ran.iter().map(|ran| ran.value.as_int()).collect();
    assert_eq!(values[0], values[1], "the two optimization levels agree");
    values[0]
}

fn refusal(i: &Interner, helpers: &[Helper<'_>], main: &str) -> String {
    match compile_and_run(i, helpers, main, Ty::I64, Opt::None) {
        Ok(ran) => panic!("expected a refusal, ran to {:?}", ran.value),
        Err(r) => r.messages.join("\n"),
    }
}

// -- Types ---------------------------------------------------------------

/// A recursive call's result is the body's own result, an `i64`; comparing
/// it with a text was compiled and trapped at run time while each recursive
/// call took a fresh instance of the undeclared result.
#[test]
fn a_recursive_result_compared_with_a_text_is_refused_at_the_comparison() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        &[helper(
            &i,
            "f",
            "if $n == 0 { 1 } else { if f($n - 1) == \"x\" { 2 } else { 3 } }",
        )],
        "f(3)",
    );
    assert_eq!(refused, "[f] type mismatch in `==`: String vs i64");
}

#[test]
fn a_recursive_result_used_at_its_own_type_runs() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        &[helper(
            &i,
            "f",
            "if $n == 0 { 1 } else { if f($n - 1) == 1 { 2 } else { 3 } }",
        )],
        "f(3)",
    );
    assert_eq!(value, 3);
}

/// Where the refusal lands depends on which member the component checks
/// first, which is the member the entry's call reaches first: each body's
/// solve settles its own decisions, so the `==` in `a` settles at a text
/// when `a` is solved before `b` joins `a`'s result, and `b`'s tail is then
/// the refusal. Either way the program is refused.
#[test]
fn a_mutually_recursive_result_compared_with_a_text_is_refused_from_either_member() {
    let i = Interner::new();
    let helpers = [
        helper(
            &i,
            "a",
            "if $n == 0 { 1 } else { if b($n - 1) == \"x\" { 2 } else { 3 } }",
        ),
        helper(&i, "b", "if $n == 0 { 0 } else { a($n - 1) }"),
    ];
    assert_eq!(
        refusal(&i, &helpers, "a(3)"),
        "[b] type mismatch: expected String, got i64"
    );
    assert_eq!(
        refusal(&i, &helpers, "b(3)"),
        "[a] type mismatch in `==`: i64 vs str"
    );
}

fn even_odd<'a>(i: &Interner) -> Vec<Helper<'a>> {
    vec![
        helper(i, "is_even", "if $n == 0 { true } else { is_odd($n - 1) }"),
        helper(i, "is_odd", "if $n == 0 { false } else { is_even($n - 1) }"),
    ]
}

#[test]
fn a_mutually_recursive_pair_with_undeclared_results_is_typed_as_one() {
    let i = Interner::new();
    let even = "if is_even(4) { 1 } else { 0 }";
    assert_eq!(integer_at_both_levels(&i, &even_odd(&i), even), 1);
    let odd = "if is_even(7) { 1 } else { 0 }";
    assert_eq!(integer_at_both_levels(&i, &even_odd(&i), odd), 0);
}

/// The member the entry's call reaches first is checked first, so each
/// entry checks the component in another order.
#[test]
fn a_result_only_a_sibling_decides_is_the_sibling_s_from_either_member() {
    let i = Interner::new();
    let helpers = [
        helper(&i, "pass", "text($n)"),
        helper(&i, "text", "if $n == 0 { \"done\" } else { pass($n - 1) }"),
    ];
    for main in [
        "if pass(3) == \"done\" { 1 } else { 0 }",
        "if text(3) == \"done\" { 1 } else { 0 }",
    ] {
        assert_eq!(integer_at_both_levels(&i, &helpers, main), 1);
    }
}

// -- Effects -------------------------------------------------------------

fn int() -> acvus_mir::ty::PolyTy {
    lift_to_poly(&Ty::I64)
}

fn local(i: &Interner, pb: &mut PolyBuilder, name: &str, source: &str) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Local(
            ParsedAst::Script(acvus_ast::parse_script(i, source).expect("the body parses")),
            Inputs::Declared,
        ),
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(i.intern("n"), int())],
            ret: Box::new(lift_declaration(&Ty::I64, pb)),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: Flows::Every.into(),
        },
    }
}

struct Body {
    name: &'static str,
    source: &'static str,
}

fn effects(i: &Interner, bodies: &[Body]) -> FxHashMap<&'static str, Effect> {
    let mut pb = PolyBuilder::new();
    let functions: Vec<Function> = bodies
        .iter()
        .map(|body| local(i, &mut pb, body.name, body.source))
        .collect();
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![Context {
            qref: QualifiedRef::root(i.intern("c")),
            ty: lift_declaration(&Ty::I64, &mut pb),
            init: None,
        }]),
        types: Freeze::default(),
        bindings: Bindings::default(),
        access: Access::Sync,
        entries: vec![],
    };
    let extracted = extract::extract(i, &graph);
    let inferred = infer::infer(i, &graph, &extracted);
    let errors: Vec<String> = inferred
        .errors()
        .into_iter()
        .flat_map(|(_, errors)| errors.iter().map(|e| e.display(i).to_string()))
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    bodies
        .iter()
        .map(|body| {
            let name = body.name;
            let outcome = &inferred.outcomes[&QualifiedRef::root(i.intern(name))];
            let Ty::Fn {
                effect: acvus_mir::ty::EffectTerm::Known(effect),
                ..
            } = &outcome.meta().ty
            else {
                panic!("{name}'s type is a function with a closed effect");
            };
            (name, effect.clone())
        })
        .collect()
}

fn only_c(i: &Interner) -> BTreeSet<QualifiedRef> {
    BTreeSet::from([QualifiedRef::root(i.intern("c"))])
}

#[test]
fn a_read_reached_through_a_cycle_of_two_is_every_member_s() {
    let i = Interner::new();
    let effects = effects(
        &i,
        &[
            Body {
                name: "f",
                source: "if $n == 0 { 0 } else { g($n - 1) }",
            },
            Body {
                name: "g",
                source: "if $n == 0 { @c } else { f($n - 1) }",
            },
            Body {
                name: "k",
                source: "g($n)",
            },
        ],
    );
    for name in ["f", "g", "k"] {
        assert_eq!(effects[name].reads, only_c(&i), "{name}: {:?}", effects[name]);
        assert!(effects[name].writes.is_empty(), "{name}: {:?}", effects[name]);
    }
}

#[test]
fn a_write_reached_through_a_cycle_of_three_is_every_member_s() {
    let i = Interner::new();
    let effects = effects(
        &i,
        &[
            Body {
                name: "f",
                source: "if $n == 0 { 0 } else { g($n - 1) }",
            },
            Body {
                name: "g",
                source: "if $n == 0 { 0 } else { h($n - 1) }",
            },
            Body {
                name: "h",
                source: "if $n == 0 { @c = 1; 0 } else { f($n - 1) }",
            },
        ],
    );
    for name in ["f", "g", "h"] {
        assert_eq!(effects[name].writes, only_c(&i), "{name}: {:?}", effects[name]);
    }
}

#[test]
fn a_read_reached_through_a_cycle_of_three_is_every_member_s() {
    let i = Interner::new();
    let effects = effects(
        &i,
        &[
            Body {
                name: "f",
                source: "if $n == 0 { 0 } else { g($n - 1) }",
            },
            Body {
                name: "g",
                source: "if $n == 0 { 0 } else { h($n - 1) }",
            },
            Body {
                name: "h",
                source: "if $n == 0 { @c } else { f($n - 1) }",
            },
        ],
    );
    for name in ["f", "g", "h"] {
        assert_eq!(effects[name].reads, only_c(&i), "{name}: {:?}", effects[name]);
    }
}

// -- The bracket -----------------------------------------------------------

/// `f` touches no context itself, and `g`, which only `f` calls, writes
/// `@c`. RFC-0025 rule 2 brackets the entry's call to `f` by `f`'s summary,
/// so the entry commits its `@c = 1` before the call and fetches the
/// callee's 7 after it.
#[test]
fn a_write_reached_only_through_recursion_is_bracketed_at_the_caller() {
    let i = Interner::new();
    let helpers = [
        helper(&i, "f", "if $n == 0 { 0 } else { g($n - 1) }"),
        helper(&i, "g", "if $n == 0 { @c = 7; 0 } else { f($n - 1) }"),
    ];
    for ran in ran_at_both_levels(&i, &helpers, "@c = 1;\nlet r = f(3);\n@c + r", Ty::I64) {
        assert_eq!(ran.value.as_int(), 7);
        assert_eq!(
            ran.committed,
            vec![Committed {
                context: "c".to_string(),
                value: 7
            }]
        );
    }
}
