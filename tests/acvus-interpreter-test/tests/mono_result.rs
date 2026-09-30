//! A `Result` of a member type crosses a `Monomorphize` member by value, at
//! the parameter and at the result: the script picks the instance by the
//! argument's type, the handler reads Rust's `Result<T, E>` and writes one
//! back, and both arms reach the script as the tag they were built with.

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{Refusal, check_source, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Twice the value, as each member type defines it.
trait Twice {
    fn twice(self) -> Self;
}

impl Twice for i64 {
    fn twice(self) -> Self {
        self * 2
    }
}

impl Twice for String {
    fn twice(self) -> Self {
        format!("{self}{self}")
    }
}

#[extern_fn(effect = pure)]
fn twice_ok<A>(r: Result<A, String>) -> Result<A, String>
where
    A: acvus_extern::Monomorphize<(i64, String)> + Twice,
{
    r.map(Twice::twice)
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        fns: [twice_ok],
    });
    regs
}

fn compile_and_run(i: &Interner, source: &str, opt: Opt) -> Result<Value, Refusal> {
    let ast = ParsedAst::Script(acvus_ast::parse_script(i, source).expect("the script parses"));
    let cr = check_source(
        i,
        ast,
        &FxHashMap::default(),
        regs(),
        Ty::String,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()).expect("the seeds hold every context the run fetches"))
}

/// The text a program yields at both optimization levels. Disagreement is the
/// differential's own contract, so it fails here rather than being reported as
/// one string.
fn text_at_both_levels(i: &Interner, source: &str) -> String {
    let read = |opt| {
        let value = compile_and_run(i, source, opt)
            .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")));
        assert!(value.is_string(), "expected a String, got {value:?}");
        // SAFETY: the script's declared return type is `String`.
        unsafe { value.as_str() }.to_owned()
    };
    let full = read(Opt::Full);
    let none = read(Opt::None);
    assert_eq!(full, none, "the two optimization levels read one Result");
    full
}

fn match_both_arms(argument: &str) -> String {
    format!(
        r#"let r = t::twice_ok({argument});
           if let Ok(v) = r {{ "ok " + v.to_string() }}
           else if let Err(e) = r {{ "err " + e }}
           else {{ "unreachable".to_string() }}"#
    )
}

#[test]
fn the_i64_member_doubles_the_ok_side_and_carries_the_err_side() {
    let i = Interner::new();
    assert_eq!(text_at_both_levels(&i, &match_both_arms("Ok(21)")), "ok 42");
    assert_eq!(
        text_at_both_levels(
            &i,
            &match_both_arms(r#"if false { Ok(21) } else { Err("bad".to_string()) }"#)
        ),
        "err bad"
    );
}

#[test]
fn the_string_member_runs_its_own_instance() {
    let i = Interner::new();
    assert_eq!(
        text_at_both_levels(&i, &match_both_arms(r#"Ok("ab".to_string())"#)),
        "ok abab"
    );
}
