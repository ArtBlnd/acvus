use acvus_extern::{Bottom, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn boom(message: String) -> Bottom {
    panic!("boom: {message}")
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        fns: [boom],
    });
    regs
}

fn flag(i: &Interner, b: bool) -> Context {
    [(i.intern("c"), typed(Ty::Bool, Value::bool_(b)))]
        .into_iter()
        .collect()
}

#[tokio::test]
async fn a_branch_that_panics_leaves_the_other_branch_s_type() {
    let i = Interner::new();
    let v = run_script_mode_with_externs(
        &i,
        r#"if @c { boom("no".to_string()) } else { 41 }"#,
        flag(&i, false),
        registries(),
        Ty::I64,
    )
    .await
    .value;
    assert_eq!(v.as_int(), 41);
}

#[tokio::test]
#[should_panic(expected = "boom: no")]
async fn a_panic_stops_the_run_with_its_message() {
    let i = Interner::new();
    run_script_mode_with_externs(
        &i,
        r#"if @c { boom("no".to_string()) } else { 41 }"#,
        flag(&i, true),
        registries(),
        Ty::I64,
    )
    .await;
}

#[tokio::test]
async fn a_move_on_the_panicking_path_does_not_reach_the_code_after() {
    let i = Interner::new();
    let v = run_script_mode_with_externs(
        &i,
        r#"let s = "kept".to_string(); if @c { let t = s; boom(t) } else { 0 }; len(&s)"#,
        flag(&i, false),
        registries(),
        Ty::U64,
    )
    .await
    .value;
    assert_eq!(v.as_int(), 4);
}

#[tokio::test]
async fn a_panic_as_an_operand_is_below_the_other_operand_s_type() {
    let i = Interner::new();
    for source in [
        r#"if @c { boom("no".to_string()) + 1 } else { 41 }"#,
        r#"if @c { 1 + boom("no".to_string()) } else { 41 }"#,
        r#"if @c { -boom("no".to_string()) } else { 41 }"#,
    ] {
        let v = run_script_mode_with_externs(&i, source, flag(&i, false), registries(), Ty::I64)
            .await
            .value;
        assert_eq!(v.as_int(), 41, "{source}");
    }
}

#[tokio::test]
#[should_panic(expected = "boom: no")]
async fn a_panic_as_an_operand_stops_the_run_with_its_message() {
    let i = Interner::new();
    run_script_mode_with_externs(
        &i,
        r#"if @c { boom("no".to_string()) + 1 } else { 41 }"#,
        flag(&i, true),
        registries(),
        Ty::I64,
    )
    .await;
}

// -- An index into `[]` ----------------------------------------------------

/// Where `[][0]`'s element is written: a host function's body, a script, a
/// lambda, a script's `fn`. Each result is unused, so nothing but RFC-0042
/// rule 3 settles the element.
struct IndexPosition {
    name: &'static str,
    helper_body: Option<&'static str>,
    main: &'static str,
    levels: &'static [Opt],
}

/// A body that ends in `Diverge`, inlined at `Opt::Full`, leaves the
/// `Diverge` inside the caller's block, and the interval analysis stops
/// there; `panic(..)` in place of `[][0]` does the same. That is the
/// inliner's defect, queued beside MIR M4, so a function called by name runs
/// at `Opt::None` here.
const EMPTY_INDEX_POSITIONS: [IndexPosition; 4] = [
    IndexPosition {
        name: "host function body",
        helper_body: Some("[][0]"),
        main: "let x = f(1); 1",
        levels: &[Opt::None],
    },
    IndexPosition {
        name: "script",
        helper_body: None,
        main: "let x = [][0]; 1",
        levels: &[Opt::None, Opt::Full],
    },
    IndexPosition {
        name: "lambda",
        helper_body: None,
        main: "let f = |x| -> [][0]; let y = f(1); 1",
        levels: &[Opt::None, Opt::Full],
    },
    IndexPosition {
        name: "script fn",
        helper_body: None,
        main: "fn g(n) { [][0] } let y = g(1); 1",
        levels: &[Opt::None],
    },
];

fn run_empty_index(
    i: &Interner,
    position: &IndexPosition,
    opt: Opt,
) -> Result<Result<Value, acvus_interpreter::HostError>, Refusal> {
    let helpers: Vec<Helper<'_>> = position
        .helper_body
        .into_iter()
        .map(|source| Helper {
            name: "f",
            source,
            params: vec![acvus_mir::ty::ParamTerm::<acvus_mir::ty::Poly>::new(
                i.intern("n"),
                acvus_mir::ty::lift_to_poly(&Ty::I64),
            )],
        })
        .collect();
    let (context_types, snapshot) = split_context(i, int_context(i, "c", 0));
    let main = acvus_mir::graph::ParsedAst::Script(
        acvus_ast::parse_script(i, position.main).expect("main parses"),
    );
    let cr = check_graph(
        i,
        main,
        &helpers,
        &context_types,
        acvus_ext::std_registries::<AcvusRuntime>(),
        Ty::I64,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        i,
        cr,
        snapshot,
        std::sync::Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()))
}

/// `[]`'s element closes to `!` (RFC-0042 rule 3, RFC-0038 rule 2), a word,
/// so `[][0]` is an index in `Copy` mode (RFC-0047 rule 4) and the run traps
/// at the bound check. Checking and lowering panicked where the element
/// was left unsettled.
#[test]
fn an_index_into_an_empty_list_is_accepted_at_never_and_traps_in_every_position() {
    let i = Interner::new();
    for position in &EMPTY_INDEX_POSITIONS {
        for &opt in position.levels {
            match run_empty_index(&i, position, opt) {
                Ok(Err(acvus_interpreter::HostError::Trapped { message })) => assert_eq!(
                    message, "index out of bounds: the len is 0 but the index is 0",
                    "{} at {opt:?}",
                    position.name
                ),
                Ok(Err(other)) => panic!("{} at {opt:?} ended with {other:?}", position.name),
                Ok(Ok(value)) => panic!("{} at {opt:?} ran to {value:?}", position.name),
                Err(r) => panic!(
                    "{} at {opt:?} refused:\n  {}",
                    position.name,
                    r.messages.join("\n  ")
                ),
            }
        }
    }
}
