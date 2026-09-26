//! RFC-0100: a script declares a function with `fn`. It is visible
//! throughout its script and in no other, it captures nothing, and it is
//! typed as a lambda, one instance per call site, with the functions that
//! call each other typed as one component.

use std::sync::Arc;

use acvus_ast::Span;
use acvus_interpreter::{AcvusRuntime, Host, HostError, SequentialExecutor, Source};
use acvus_interpreter_test::{Helper, Refusal, check_graph, execute_compiled, int_context, split_context};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{ParamTerm, Poly, Ty, lift_to_poly};
use acvus_utils::Interner;

fn compile_and_run(i: &Interner, helpers: &[Helper<'_>], main: &str, opt: Opt) -> Result<i64, Refusal> {
    let (context_types, snapshot) = split_context(i, int_context(i, "c", 5));
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let compiled = check_graph(
        i,
        parsed,
        helpers,
        &context_types,
        acvus_ext::std_registries::<AcvusRuntime>(),
        Ty::I64,
        opt,
        |_| {},
    )?;
    let (_, mut interp) =
        execute_compiled(i, compiled, snapshot, Arc::new(acvus_interpreter::SequentialExecutor));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    let value = runtime
        .block_on(interp.execute())
        .expect("the seeds hold every context the run fetches");
    Ok(value.as_int())
}

fn run_at_both_levels(helpers: &[Helper<'_>], main: &str) -> i64 {
    run_in(&Interner::new(), helpers, main)
}

fn run_in(i: &Interner, helpers: &[Helper<'_>], main: &str) -> i64 {
    let values: Vec<i64> = [Opt::None, Opt::Full]
        .into_iter()
        .map(|opt| {
            compile_and_run(i, helpers, main, opt)
                .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")))
        })
        .collect();
    assert_eq!(values[0], values[1], "the two optimization levels agree");
    values[0]
}

fn host() -> Host {
    Host::new(acvus_ext::std_registries::<AcvusRuntime>())
}

fn refusals(host: Host) -> Vec<acvus_interpreter::Refusal> {
    match host.compile(SequentialExecutor) {
        Ok(_) => panic!("the program compiled"),
        Err(HostError::Refused(refusals)) => refusals,
        Err(other) => panic!("a compilation is refused with its refusals, not {other:?}"),
    }
}

fn only_refusal(host: Host) -> acvus_interpreter::Refusal {
    let mut refused = refusals(host);
    assert_eq!(refused.len(), 1, "{refused:?}");
    refused.remove(0)
}

fn span_of(source: &str, written: &str, nth: usize) -> Span {
    let start = source
        .match_indices(written)
        .nth(nth)
        .unwrap_or_else(|| panic!("`{written}` is written {} times", nth + 1))
        .0;
    Span::new(start, start + written.len())
}

/// The name the `nth` `fn` of `source` declares.
fn fn_name_of(source: &str, name: &str, nth: usize) -> Span {
    let written = span_of(source, &format!("fn {name}("), nth);
    Span::new(written.start + 3, written.start + 3 + name.len())
}

fn script_refusal(source: &str) -> acvus_interpreter::Refusal {
    only_refusal(host().entry::<(), i64>("main", Source::Script(source)))
}

#[test]
fn a_fn_is_called_before_and_after_its_declaration_and_from_every_block() {
    let main = "let a = before(1);
fn before(x) { x + 1 }
let b = if true { nested(2) } else { 0 };
if false { fn nested(x) { x * 10 } }
fn later() { before(100) }
a + b + later()";
    assert_eq!(run_at_both_levels(&[], main), 2 + 20 + 101);
}

#[test]
fn a_fn_declared_in_a_fn_is_visible_to_the_whole_script() {
    let main = "fn outer() { fn inner(x) { x + 1 } inner(1) }
outer() + inner(10)";
    assert_eq!(run_at_both_levels(&[], main), 2 + 11);
}

#[test]
fn a_fn_is_callable_from_no_other_script() {
    let refused = only_refusal(
        host()
            .entry::<(), i64>("a", Source::Script("fn f() { 1 }\nf()"))
            .entry::<(), i64>("b", Source::Script("f()")),
    );
    assert_eq!(refused.message, "undefined function `f`");
    assert_eq!(refused.span, Some(Span::new(0, 3)));
}

#[test]
fn two_fns_of_one_name_in_different_blocks_are_refused() {
    let source = "fn f() { 1 }\nif true { fn f() { 2 } }\nf()";
    let refused = script_refusal(source);
    assert_eq!(refused.message, "the function `f` is declared twice in this script");
    assert_eq!(refused.span, Some(fn_name_of(source, "f", 1)));
    assert_eq!(refused.labels.len(), 1);
    assert_eq!(refused.labels[0].span, Some(fn_name_of(source, "f", 0)));
    assert_eq!(refused.labels[0].text, "`f` is first declared here");
}

#[test]
fn a_fn_that_shadows_an_extern_is_refused() {
    let source = "fn len(x) { 1 }\nlen(2)";
    let refused = script_refusal(source);
    assert_eq!(
        refused.message,
        "the `fn` `len` would shadow `array::len`, `deque::len`, `map::len`, `set::len`, \
         `slice::len`, `string::len`, `vec::len`, which the host declares and a script calls as `len`"
    );
    assert_eq!(refused.span, Some(span_of(source, "len", 0)));
}

#[test]
fn a_fn_that_shadows_another_entry_is_refused() {
    let source = "fn other() { 1 }\nother()";
    let refused = only_refusal(
        host()
            .entry::<(), i64>("main", Source::Script(source))
            .entry::<(), i64>("other", Source::Script("2")),
    );
    assert_eq!(
        refused.message,
        "the `fn` `other` would shadow `other`, which the host declares and a script calls as `other`"
    );
    assert_eq!(refused.span, Some(span_of(source, "other", 0)));
}

#[test]
fn a_template_declares_no_fn() {
    let refused = only_refusal(host().entry::<(), String>("main", Source::Template("% fn f() { 1 }\nx\n")));
    assert_eq!(
        refused.message,
        "only a script (`.acvus`) declares a `fn`; a template declares none"
    );
}

#[test]
fn a_template_declares_no_fn_in_a_block_either() {
    let source = "{{ { fn f() { 1 } f() } }}\n";
    let refused = refusals(host().entry::<(), String>("main", Source::Template(source)));
    assert!(
        refused.iter().any(|refusal| refusal.message
            == "only a script (`.acvus`) declares a `fn`; a template declares none"
            && refusal.span == Some(fn_name_of(source, "f", 0))),
        "{refused:?}"
    );
}

#[test]
fn a_fn_reading_a_local_of_the_script_is_refused_naming_where_it_is_declared() {
    let source = "let y = 1;\nfn f() { y }\nf()";
    let refused = script_refusal(source);
    assert_eq!(
        refused.message,
        "a `fn` captures nothing: its body reads `y`, a local of the script"
    );
    assert_eq!(refused.span, Some(span_of(source, "y", 1)));
    assert_eq!(refused.labels.len(), 1);
    assert_eq!(refused.labels[0].span, Some(span_of(source, "y", 0)));
    assert_eq!(refused.labels[0].text, "`y` is declared here, outside the `fn`");
}

#[test]
fn a_local_declared_only_after_the_fn_is_still_refused() {
    let source = "fn f() { y }\nlet y = 1;\nf()";
    let refused = script_refusal(source);
    assert_eq!(
        refused.message,
        "a `fn` captures nothing: its body reads `y`, a local of the script"
    );
    assert_eq!(refused.span, Some(span_of(source, "y", 0)));
    assert_eq!(refused.labels[0].span, Some(span_of(source, "y", 1)));
}

#[test]
fn a_fn_reading_an_input_is_refused() {
    let source = "fn f() { $x }\nf()";
    let refused = script_refusal(source);
    assert_eq!(
        refused.message,
        "a `fn` captures nothing: its body reads `$x`, an input of the script; pass it as an argument"
    );
    assert_eq!(refused.span, Some(span_of(source, "$x", 0)));
}

#[test]
fn a_fn_reading_a_context_is_refused() {
    let source = "fn f() { @c }\nf()";
    let refused = script_refusal(source);
    assert_eq!(
        refused.message,
        "a `fn` captures nothing: its body names `@c`, a context the host declares; pass it as an argument"
    );
    assert_eq!(refused.span, Some(span_of(source, "@c", 0)));
}

#[test]
fn a_parameter_read_as_an_input_is_refused() {
    let refused = script_refusal("fn f(n) { $n }\nf(1)");
    assert_eq!(
        refused.message,
        "a `fn` captures nothing: its body reads `$n`, an input of the script; pass it as an argument"
    );
}

#[test]
fn a_fault_in_a_fn_called_twice_is_refused_once() {
    let source = "fn f(x) { x + true }\nf(1) + f(2)";
    let refused = script_refusal(source);
    assert_eq!(refused.message, "type mismatch in `+`: i64 vs Bool");
}

#[test]
fn a_fn_calls_other_fns_externs_and_host_functions() {
    let i = Interner::new();
    let helper = Helper {
        name: "h",
        source: "$n * 3",
        params: vec![ParamTerm::<Poly>::new(i.intern("n"), lift_to_poly(&Ty::I64))],
    };
    let main = "fn total(xs) { size(xs) + h(2) }
fn size(xs) { xs.len() as i64 }
total([1, 2, 3])";
    assert_eq!(run_in(&i, &[helper], main), 3 + 6);
}

#[test]
fn a_lambda_still_captures() {
    assert_eq!(run_at_both_levels(&[], "let k = 10;\nlet add = |v| -> v + k;\nadd(1)"), 11);
}

#[test]
fn a_fn_is_used_at_two_types() {
    let main = "fn id(x) { x }
let s = id(\"s\");
if s == \"s\" { id(41) + 1 } else { 0 }";
    assert_eq!(run_at_both_levels(&[], main), 42);
}

#[test]
fn each_call_site_s_instance_takes_the_width_its_call_gives() {
    assert_eq!(run_at_both_levels(&[], "fn inc(x) { x + 1 }\n(inc(1u8) + 1u8) as i64"), 3);
}

#[test]
fn a_recursive_fn_runs() {
    let main = "fn fact(n) { if n == 0 { 1 } else { n * fact(n - 1) } }\nfact(10)";
    assert_eq!(run_at_both_levels(&[], main), 3_628_800);
}

#[test]
fn mutually_recursive_fns_run() {
    let main = "fn even(n) { if n == 0 { true } else { odd(n - 1) } }
fn odd(n) { if n == 0 { false } else { even(n - 1) } }
if even(10) && odd(7) && !even(3) { 1 } else { 0 }";
    assert_eq!(run_at_both_levels(&[], main), 1);
}

#[test]
fn a_recursive_result_compared_with_a_text_is_refused() {
    let source = "fn f(n) { if n == 0 { 1 } else { if f(n - 1) == \"x\" { 2 } else { 3 } } }\nf(3)";
    let refused = script_refusal(source);
    assert!(refused.message.starts_with("type mismatch"), "{}", refused.message);
}
