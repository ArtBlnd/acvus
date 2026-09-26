//! RFC-0106 rule 3: a source nested exactly to `NESTING_MAX`, for each kind
//! of level, compiles on a thread of 256 KiB, since every walk of the
//! compiler enters its recursion through `acvus_utils::grow`, and runs to its
//! value.
//!
//! The run is not on the small thread. The machine keeps a headroom below
//! each call for a trap (RFC-0100 rule 5), and a debug build's headroom is
//! more than a 256 KiB thread holds, so a body that calls a closure traps at
//! its first call there, at any nesting: observed 2026-09-26 for the lambda
//! and pattern sources. It runs on a thread of `std`'s default stack instead,
//! through the host's typed entry (RFC-0090), whose `Output` lends the result
//! as the Rust type the entry declares. That entry compiles the source again
//! and prepares it, so the preparation is on the run's thread too, where it
//! was before the host carried the run; preparing the lambda and pattern
//! sources overflows 256 KiB in the loan analysis's walk of their nested
//! types: observed 2026-09-26.
//!
//! The sources run in a child process: an overflow aborts the whole process,
//! so a walk left outside `grow` fails the one test that spawned the child
//! rather than ending the runner.

use std::process::Command;

use acvus_ast::error::ParseErrorKind;
use acvus_ast::NESTING_MAX;
use acvus_interpreter::{AcvusRuntime, Host, MemoryStorage, SequentialExecutor, Source};
use acvus_interpreter_test::check_source;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

const CHILD: &str = "nesting_bound::child::";

const CHILD_TESTS: usize = 7;

const SMALL_STACK: usize = 256 << 10;

const RUN_STACK: usize = 2 << 20;

const MAX: usize = NESTING_MAX as usize;

const ENTRY: &str = "main";

#[derive(Clone, Copy)]
enum Form {
    Script,
    Template,
}

struct Kind {
    form: Form,
    source: fn(usize) -> String,
    value: fn(usize) -> Expected,
}

#[derive(Debug, PartialEq)]
enum Expected {
    Int(i64),
    Text(String),
}

fn parens(levels: usize) -> String {
    format!("{}1{}", "(".repeat(levels - 1), ")".repeat(levels - 1))
}

fn chain(levels: usize) -> String {
    format!("1{}", " + 1".repeat(levels - 1))
}

fn blocks(levels: usize) -> String {
    format!("{}1{}", "{ ".repeat(levels - 1), " }".repeat(levels - 1))
}

fn statements(levels: usize) -> String {
    let (innermost, below) = match levels % 2 == 0 {
        true => ("(1)", 2),
        false => ("1", 1),
    };
    let count = (levels - below) / 2;
    format!("{}{innermost}{}", "{ let a = ".repeat(count), "; a }".repeat(count))
}

fn lambdas(levels: usize) -> String {
    let count = levels - 2;
    format!(
        "let f = {}x; f{}",
        "|x| -> ".repeat(count),
        "(1)".repeat(count)
    )
}

fn patterns(levels: usize) -> String {
    let count = levels - 2;
    let some = |inner: &str| format!("{}{inner}{}", "Some(".repeat(count), ")".repeat(count));
    format!("match {} {{ {} => x, _ => 0 }}", some("1"), some("x"))
}

fn sections(levels: usize) -> String {
    let count = (levels - 2) / 2;
    let line = match levels % 2 == 0 {
        true => "x\n",
        false => "{{ (\"x\") }}\n",
    };
    format!("{}{line}{}", "% if true\n".repeat(count), "% end\n".repeat(count))
}

fn kinds() -> [(&'static str, Kind); CHILD_TESTS] {
    [
        ("parens", Kind { form: Form::Script, source: parens, value: |_| Expected::Int(1) }),
        (
            "chain",
            Kind {
                form: Form::Script,
                source: chain,
                value: |levels| Expected::Int(i64::try_from(levels).expect("a small count")),
            },
        ),
        ("blocks", Kind { form: Form::Script, source: blocks, value: |_| Expected::Int(1) }),
        (
            "statements",
            Kind { form: Form::Script, source: statements, value: |_| Expected::Int(1) },
        ),
        ("lambdas", Kind { form: Form::Script, source: lambdas, value: |_| Expected::Int(1) }),
        ("patterns", Kind { form: Form::Script, source: patterns, value: |_| Expected::Int(1) }),
        (
            "sections",
            Kind {
                form: Form::Template,
                source: sections,
                value: |_| Expected::Text("x\n".to_owned()),
            },
        ),
    ]
}

fn kind(name: &str) -> Kind {
    kinds()
        .into_iter()
        .find_map(|(named, kind)| (named == name).then_some(kind))
        .expect("a kind of level")
}

fn parsed(interner: &Interner, form: Form, source: &str) -> Result<ParsedAst, Vec<ParseErrorKind>> {
    let kinds = |errors: Vec<acvus_ast::ParseError>| -> Vec<ParseErrorKind> {
        errors.into_iter().map(|error| error.kind).collect()
    };
    match form {
        Form::Script => acvus_ast::parse_script(interner, source)
            .map(ParsedAst::Script)
            .map_err(|recovered| kinds(recovered.errors)),
        Form::Template => acvus_ast::parse_template(interner, source)
            .map(ParsedAst::Template)
            .map_err(|recovered| kinds(recovered.errors)),
    }
}

fn checked(interner: &Interner, form: Form, source: &str, opt: Opt) {
    let ast = parsed(interner, form, source)
        .unwrap_or_else(|errors| panic!("the source is refused: {errors:?}"));
    let ret = match form {
        Form::Script => Ty::I64,
        Form::Template => Ty::String,
    };
    check_source(
        interner,
        ast,
        &FxHashMap::default(),
        acvus_ext::std_registries::<AcvusRuntime>(),
        ret,
        opt,
        |_| {},
    )
    .unwrap_or_else(|refusal| panic!("{opt:?} refused:\n  {}", refusal.messages.join("\n  ")));
}

fn ran(form: Form, source: &str, opt: Opt) -> Expected {
    let host = Host::new(acvus_ext::std_registries::<AcvusRuntime>()).opt(opt);
    let host = match form {
        Form::Script => host.entry::<(), i64>(ENTRY, Source::Script(source)),
        Form::Template => host.entry::<(), String>(ENTRY, Source::Template(source)),
    };
    let program = host
        .compile(SequentialExecutor)
        .unwrap_or_else(|error| panic!("the host refused at {opt:?}: {error:?}"));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    runtime.block_on(program.scope(async |s| {
        let mut storage = MemoryStorage::new();
        let mut page = s.open(&mut storage);
        match form {
            Form::Script => {
                let entry = s.entry::<(), i64>(ENTRY).expect("the entry returns `i64`");
                let output = entry
                    .run(&mut page, ())
                    .await
                    .unwrap_or_else(|error| panic!("the run ended with {error:?}"));
                Expected::Int(output.with(|n: &i64| *n).expect("the result is an `i64`"))
            }
            Form::Template => {
                let entry = s.entry::<(), String>(ENTRY).expect("the entry returns `String`");
                let output = entry
                    .run(&mut page, ())
                    .await
                    .unwrap_or_else(|error| panic!("the run ended with {error:?}"));
                Expected::Text(output.with(|text: &str| text.to_owned()).expect("the result is a `String`"))
            }
        }
    }))
}

fn on_thread<T, F>(stack: usize, work: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(work)
        .expect("the thread starts")
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}

fn at_the_bound(name: &str) {
    let kind = kind(name);
    let interner = Interner::new();
    let past = (kind.source)(MAX + 1);
    assert_eq!(
        parsed(&interner, kind.form, &past).err(),
        Some(vec![ParseErrorKind::NestingTooDeep { max: NESTING_MAX }]),
        "{name}: a level past the bound is refused"
    );
    let source = (kind.source)(MAX);
    for opt in [Opt::None, Opt::Full] {
        let form = kind.form;
        let at = source.clone();
        let compiling = interner.clone();
        on_thread(SMALL_STACK, move || checked(&compiling, form, &at, opt));
        let at = source.clone();
        let value = on_thread(RUN_STACK, move || ran(form, &at, opt));
        assert_eq!(value, (kind.value)(MAX), "{name} at {opt:?}");
    }
}

#[test]
fn every_kind_at_the_bound_compiles_on_a_small_thread_in_a_child_process() {
    let exe = std::env::current_exe().expect("the test binary's own path");
    let out = Command::new(exe)
        .args([CHILD, "--ignored", "--test-threads=1"])
        .output()
        .expect("the test binary runs as a child");
    let stdout = String::from_utf8(out.stdout).expect("libtest writes UTF-8");
    let stderr = String::from_utf8(out.stderr).expect("libtest writes UTF-8");
    assert!(out.status.success(), "the child ended with {}:\n{stdout}\n{stderr}", out.status);
    let passed = format!("test result: ok. {CHILD_TESTS} passed");
    assert!(stdout.contains(&passed), "the child ran other than {CHILD_TESTS} tests:\n{stdout}");
}

mod child {
    use super::*;

    #[test]
    #[ignore = "may exhaust a thread's stack; run by the parent test in a child process"]
    fn parens() {
        at_the_bound("parens");
    }

    #[test]
    #[ignore = "may exhaust a thread's stack; run by the parent test in a child process"]
    fn chain() {
        at_the_bound("chain");
    }

    #[test]
    #[ignore = "may exhaust a thread's stack; run by the parent test in a child process"]
    fn blocks() {
        at_the_bound("blocks");
    }

    #[test]
    #[ignore = "may exhaust a thread's stack; run by the parent test in a child process"]
    fn statements() {
        at_the_bound("statements");
    }

    #[test]
    #[ignore = "may exhaust a thread's stack; run by the parent test in a child process"]
    fn lambdas() {
        at_the_bound("lambdas");
    }

    #[test]
    #[ignore = "may exhaust a thread's stack; run by the parent test in a child process"]
    fn patterns() {
        at_the_bound("patterns");
    }

    #[test]
    #[ignore = "may exhaust a thread's stack; run by the parent test in a child process"]
    fn sections() {
        at_the_bound("sections");
    }
}
