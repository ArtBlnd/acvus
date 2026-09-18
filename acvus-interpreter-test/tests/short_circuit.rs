//! RFC-0020: `a && b` is `if a { b } else { false }` and `a || b` is
//! `if a { true } else { b }`. The right operand is an extern whose call
//! is observable — it panics, or it counts — so a run says whether the
//! operand was evaluated, not only what the operator returned.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn boom() -> bool {
    panic!("the right operand of a decided connective was evaluated")
}

#[derive(Default)]
struct Calls(AtomicUsize);

impl Calls {
    fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

#[extern_fn(effect = opaque)]
fn counted(#[state] calls: &Arc<Calls>, value: bool) -> bool {
    calls.0.fetch_add(1, Ordering::SeqCst);
    value
}

#[derive(Clone, Copy)]
struct Operands {
    left: bool,
    right: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    value: bool,
    calls: usize,
}

fn boom_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [boom],
    }
}

fn counted_registry(calls: &Arc<Calls>) -> Registry<AcvusRuntime> {
    let calls = Arc::clone(calls);
    extern_registry! {
        ns: "t",
        fns: [counted(Arc::clone(&calls))],
    }
}

fn context_of(i: &Interner, operands: Operands) -> Context {
    [
        (
            i.intern("left"),
            typed(Ty::Bool, Value::bool_(operands.left)),
        ),
        (
            i.intern("right"),
            typed(Ty::Bool, Value::bool_(operands.right)),
        ),
    ]
    .into_iter()
    .collect()
}

async fn run_with_boom(source: &str, left: bool) -> bool {
    let i = Interner::new();
    let context = context_of(&i, Operands { left, right: false });
    run_script_with_externs(&i, source, context, vec![boom_registry()])
        .await
        .value
        .as_bool()
}

async fn run_with_counter(source: &str, operands: Operands) -> Outcome {
    let i = Interner::new();
    let calls = Arc::new(Calls::default());
    let context = context_of(&i, operands);
    let value = run_script_with_externs(&i, source, context, vec![counted_registry(&calls)])
        .await
        .value
        .as_bool();
    Outcome {
        value,
        calls: calls.count(),
    }
}

#[tokio::test]
async fn a_false_left_operand_of_and_decides_without_the_right() {
    assert!(!run_with_boom("@left && boom()", false).await);
}

#[tokio::test]
async fn a_true_left_operand_of_or_decides_without_the_right() {
    assert!(run_with_boom("@left || boom()", true).await);
}

#[tokio::test]
async fn a_true_left_operand_of_and_takes_the_right() {
    let source = "@left && counted(@right)";
    assert_eq!(
        run_with_counter(
            source,
            Operands {
                left: true,
                right: true
            }
        )
        .await,
        Outcome {
            value: true,
            calls: 1
        }
    );
    assert_eq!(
        run_with_counter(
            source,
            Operands {
                left: true,
                right: false
            }
        )
        .await,
        Outcome {
            value: false,
            calls: 1
        }
    );
}

#[tokio::test]
async fn a_false_left_operand_of_or_takes_the_right() {
    let source = "@left || counted(@right)";
    assert_eq!(
        run_with_counter(
            source,
            Operands {
                left: false,
                right: true
            }
        )
        .await,
        Outcome {
            value: true,
            calls: 1
        }
    );
    assert_eq!(
        run_with_counter(
            source,
            Operands {
                left: false,
                right: false
            }
        )
        .await,
        Outcome {
            value: false,
            calls: 1
        }
    );
}

#[tokio::test]
async fn a_decided_connective_never_calls_the_right_operand() {
    assert_eq!(
        run_with_counter(
            "@left && counted(@right)",
            Operands {
                left: false,
                right: true
            }
        )
        .await,
        Outcome {
            value: false,
            calls: 0
        }
    );
    assert_eq!(
        run_with_counter(
            "@left || counted(@right)",
            Operands {
                left: true,
                right: true
            }
        )
        .await,
        Outcome {
            value: true,
            calls: 0
        }
    );
}
