//! RFC-0071 rule 4 at the host's contract: a `$` is one value shared by the
//! graph, so a call to a body that reads `$n` passes the caller's `$n`, and
//! an entry whose reachable code reads a `$` its inputs do not declare is
//! refused at compile time.

use acvus_extern::TyArg;
use acvus_interpreter::{AcvusRuntime, Host, HostError, MemoryStorage, Program, SequentialExecutor, Source};

#[derive(TyArg)]
pub struct N {
    n: i64,
}

#[derive(TyArg)]
pub struct Lang {
    lang: String,
}

fn host() -> Host {
    Host::new(acvus_ext::std_registries::<AcvusRuntime>())
}

fn compiled(host: Host) -> Program {
    match host.compile(SequentialExecutor) {
        Ok(program) => program,
        Err(refusals) => panic!("the program is refused: {refusals:?}"),
    }
}

fn refusal_messages(host: Host) -> Vec<String> {
    match host.compile(SequentialExecutor) {
        Ok(_) => panic!("the program compiled"),
        Err(HostError::Refused(refusals)) => refusals
            .into_iter()
            .map(|refusal| refusal.message)
            .collect(),
        Err(other) => panic!("a compilation is refused with its refusals, not {other:?}"),
    }
}

#[tokio::test]
async fn a_callee_reads_the_input_its_caller_was_run_with() {
    let program = compiled(
        host()
            .entry::<N, i64>("main", Source::Script("callee()"))
            .entry::<N, i64>("callee", Source::Script("$n + 1")),
    );
    let n = program
        .scope(async |s| {
            let mut storage = MemoryStorage::new();
            let mut page = s.open(&mut storage);
            let entry = s
                .entry::<N, i64>("main")
                .expect("`main` takes `N` and returns `i64`");
            let output = entry
                .run(&mut page, N { n: 41 })
                .await
                .expect("the graph has no context");
            output.with(|n: &i64| *n).expect("an `i64`")
        })
        .await;
    assert_eq!(n, 42);
}

#[tokio::test]
async fn a_template_s_callee_writes_the_lang_the_template_was_run_with() {
    let program = compiled(
        host()
            .entry::<Lang, String>("main", Source::Template("rules: {{ rules() }}\\\n"))
            .entry::<Lang, String>("rules", Source::Template("answer in {{ &$lang }}\\\n")),
    );
    let text = program
        .scope(async |s| {
            let mut storage = MemoryStorage::new();
            let mut page = s.open(&mut storage);
            let entry = s
                .entry::<Lang, String>("main")
                .expect("`main` takes `Lang` and returns `String`");
            let output = entry
                .run(
                    &mut page,
                    Lang {
                        lang: "ko".to_owned(),
                    },
                )
                .await
                .expect("the graph has no context");
            output.with(|s: &str| s.to_owned()).expect("a `String`")
        })
        .await;
    assert_eq!(text, "rules: answer in ko");
}

#[test]
fn an_entry_declaring_no_inputs_whose_callee_reads_one_is_refused_naming_both() {
    let messages = refusal_messages(
        host()
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
