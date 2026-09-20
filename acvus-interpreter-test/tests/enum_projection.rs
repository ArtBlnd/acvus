//! RFC-0050 rule 6, the enum half, at its contract: a script lends an enum,
//! the handler reads the arm the tag names and writes both words back, and
//! the script observes the write.

use acvus_extern::{Registry, Runtime, TyArg, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::{Context, run_script_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[derive(TyArg)]
#[projection]
pub enum Step {
    Done,
    Left(i64),
}

/// Counts the lent step down by one and closes it at zero, which is the two
/// halves of `StepMut`: `arms` writes the payload where it lies, and `set`
/// rewrites `[tag, payload]`.
#[extern_fn(effect = pure)]
fn advance<Rt>(rt: &Rt, s: StepMut<'_, Rt>) -> i64
where
    Rt: Runtime,
{
    let mut s = s;
    let left = match s.arms() {
        StepArms::Done => 0,
        StepArms::Left(n) => {
            *n -= 1;
            *n
        }
    };
    if left == 0 {
        s.set(Step::Done);
    }
    left
}

/// Reads the arm and the payload without naming a runtime: a shared
/// projection takes none.
#[extern_fn(effect = pure)]
fn left_of(s: StepRef<'_>) -> i64 {
    match s {
        StepRef::Done => 0,
        StepRef::Left(n) => *n,
    }
}

fn registry() -> Vec<Registry<AcvusRuntime>> {
    vec![extern_registry! {
        ns: "t",
        fns: [advance, left_of],
    }]
}

async fn int(source: &str) -> i64 {
    let i = Interner::new();
    let ran = run_script_with_externs(&i, source, Context::default(), registry(), Ty::I64).await;
    ran.value.as_int()
}

/// The two variants meet at the `if`, so the lent value's type is the whole
/// enum the declaration names.
const STEP: &str = "let s = if 0 < 1 { Step::Left(3) } else { Step::Done };";

#[tokio::test]
async fn a_shared_projection_reads_the_arm_the_tag_names() {
    assert_eq!(int(&format!("{STEP} left_of(&s)")).await, 3);
    assert_eq!(
        int("let s = if 1 < 0 { Step::Left(3) } else { Step::Done }; left_of(&s)").await,
        0
    );
}

#[tokio::test]
async fn the_script_reads_back_the_payload_the_handler_wrote() {
    assert_eq!(
        int(&format!("{STEP} advance(&mut s); left_of(&s)")).await,
        2
    );
}

#[tokio::test]
async fn the_script_reads_back_the_tag_set_wrote() {
    let source = "let s = if 0 < 1 { Step::Left(1) } else { Step::Done }; \
                  advance(&mut s); \
                  match s { Step::Done => { 10 }, Step::Left(n) => { n } }";
    assert_eq!(int(source).await, 10);
}

/// Each call is a call site of its own, and each site's table is filled at
/// `prepare` from that site's settled type. Three calls on one value walk
/// `Left(3)` down to `Done`.
#[tokio::test]
async fn every_call_site_carries_its_own_table() {
    let source = "let s = if 0 < 1 { Step::Left(3) } else { Step::Done }; \
                  let a = advance(&mut s); let b = advance(&mut s); let c = advance(&mut s); \
                  match s { Step::Done => { a * 100 + b * 10 + c }, Step::Left(n) => { n } }";
    assert_eq!(int(source).await, 210);
}
