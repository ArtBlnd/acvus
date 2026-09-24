//! RFC-0071 rule 4 at the host's contract: a `$` is one value shared by the
//! graph, so a call to a body that reads `$n` passes the caller's `$n`, and
//! an entry whose reachable code reads a `$` its inputs do not declare is
//! refused at compile time.

use std::sync::Arc;

use acvus_extern::TyArg;
use acvus_interpreter::{AcvusRuntime, Host, InMemoryContext, Program, SequentialExecutor, Source};
use acvus_utils::Interner;

#[derive(TyArg)]
pub struct N {
    n: i64,
}

#[derive(TyArg)]
pub struct Lang {
    lang: String,
}

fn host(i: &Interner) -> Host {
    Host::new(i, acvus_ext::std_registries::<AcvusRuntime>())
}

fn compiled(host: Host) -> Program {
    match host.compile(Arc::new(SequentialExecutor)) {
        Ok(program) => program,
        Err(refusals) => panic!("the program is refused: {refusals:?}"),
    }
}

fn refusal_messages(host: Host) -> Vec<String> {
    match host.compile(Arc::new(SequentialExecutor)) {
        Ok(_) => panic!("the program compiled"),
        Err(refusals) => refusals
            .into_iter()
            .map(|refusal| refusal.message)
            .collect(),
    }
}

#[tokio::test]
async fn a_callee_reads_the_input_its_caller_was_run_with() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<N, i64>("main", Source::Script("callee()"))
            .entry::<N, i64>("callee", Source::Script("$n + 1")),
    );
    let page = Arc::new(InMemoryContext::of(program.contexts()));
    let entry = program
        .entry::<N, i64>("main")
        .expect("`main` takes `N` and returns `i64`");
    let output = entry
        .run(&page, N { n: 41 })
        .await
        .expect("the graph has no context");
    assert_eq!(output.with(|n: &i64| *n).expect("an `i64`"), 42);
}

#[tokio::test]
async fn a_template_s_callee_writes_the_lang_the_template_was_run_with() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<Lang, String>("main", Source::Template("rules: {{ rules() }}\\\n"))
            .entry::<Lang, String>("rules", Source::Template("answer in {{ &$lang }}\\\n")),
    );
    let page = Arc::new(InMemoryContext::of(program.contexts()));
    let entry = program
        .entry::<Lang, String>("main")
        .expect("`main` takes `Lang` and returns `String`");
    let output = entry
        .run(
            &page,
            Lang {
                lang: "ko".to_owned(),
            },
        )
        .await
        .expect("the graph has no context");
    assert_eq!(
        output.with(|s: &str| s.to_owned()).expect("a `String`"),
        "rules: answer in ko"
    );
}

#[test]
fn an_entry_declaring_no_inputs_whose_callee_reads_one_is_refused_naming_both() {
    let i = Interner::new();
    let messages = refusal_messages(
        host(&i)
            .entry::<(), i64>("main", Source::Script("callee()"))
            .entry::<N, i64>("callee", Source::Script("$n + 1")),
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("`$n`") && message.contains("`callee`")),
        "{messages:?}"
    );
}
