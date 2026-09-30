//! A handler's view of the arguments it names only by type variable
//! (RFC-0097 rule 1): each is lent to a closure whose parameter type is the
//! one the checker settled at the call site, and `None` at any other.

use std::sync::Mutex;

use acvus_extern::{Args, Registry, Runtime, TyArg, Var, extern_fn, extern_registry, kind};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[derive(TyArg)]
#[projection]
pub struct Point {
    x: i64,
    y: i64,
}

#[extern_fn(effect = pure)]
fn point(x: i64, y: i64) -> Point {
    Point { x, y }
}

#[derive(Debug, PartialEq)]
struct Xy {
    x: i64,
    y: i64,
}

#[derive(Debug, PartialEq)]
struct Seen {
    n: Option<i64>,
    s: Option<String>,
    p: Option<Xy>,
    n_as_string: Option<String>,
    p_as_i64: Option<i64>,
    past_the_end: Option<i64>,
    n_after_the_write: Option<i64>,
    s_after_the_write: Option<String>,
}

static SEEN: Mutex<Vec<Seen>> = Mutex::new(Vec::new());

#[extern_fn(effect = opaque)]
fn inspect<A, B, C, R>(mut args: Args<'_, (A, B, C), R>) -> i64
where
    A: Var<kind::Type>,
    B: Var<kind::Type>,
    C: Var<kind::Type>,
    R: Runtime,
{
    let n = args.with(0, |n: &i64| *n);
    let s = args.with(1, |s: &String| s.clone());
    let p = args.with(2, |p: PointRef<'_>| Xy { x: *p.x, y: *p.y });
    let n_as_string = args.with(0, |s: &String| s.clone());
    let p_as_i64 = args.with(2, |n: &i64| *n);
    let past_the_end = args.with(3, |n: &i64| *n);
    args.with_mut(0, |n: &mut i64| *n += 1).expect("argument 0 is an `i64`");
    args.with_mut(1, |s: &mut String| s.push('!')).expect("argument 1 is a `String`");
    let n_after_the_write = args.with(0, |n: &i64| *n);
    let s_after_the_write = args.with(1, |s: &String| s.clone());
    SEEN.lock().expect("no test panicked holding the lock").push(Seen {
        n,
        s,
        p,
        n_as_string,
        p_as_i64,
        past_the_end,
        n_after_the_write,
        s_after_the_write,
    });
    0
}

#[extern_fn(effect = pure)]
fn count_three<A, B, C, R>(args: Args<'_, (A, B, C), R>) -> i64
where
    A: Var<kind::Type>,
    B: Var<kind::Type>,
    C: Var<kind::Type>,
    R: Runtime,
{
    args.len() as i64
}

#[extern_fn(effect = pure)]
fn count_one<A, R>(args: Args<'_, (A,), R>) -> i64
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.len() as i64
}

#[extern_fn(effect = pure)]
fn read_one<A, R>(args: Args<'_, (A,), R>) -> Option<i64>
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.with(0, |n: &i64| *n)
}

#[extern_fn(effect = opaque)]
fn bump_one<A, R>(mut args: Args<'_, (A,), R>) -> Option<i64>
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.with_mut(0, |n: &mut i64| {
        *n += 1;
        *n
    })
}

#[extern_fn(effect = pure)]
async fn read_one_awaited<A, R>(args: Args<'_, (A,), R>) -> Option<i64>
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.with(0, |n: &i64| *n)
}

#[extern_fn(heavy, effect = pure)]
fn read_one_offloaded<A, R>(args: Args<'_, (A,), R>) -> Option<i64>
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.with(0, |n: &i64| *n)
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [
            point,
            inspect,
            count_three,
            count_one,
            read_one,
            bump_one,
            read_one_awaited,
            read_one_offloaded,
        ],
    }
}

async fn run(i: &Interner, source: &str, ret: Ty) -> Value {
    let mut registries = acvus_ext::std_registries();
    registries.push(registry());
    run_script_with_externs(i, source, Context::default(), registries, ret)
        .await
        .value
}

/// The `i64` an `Option<i64>` script result holds, or `None`.
async fn run_optional(i: &Interner, source: &str) -> Option<i64> {
    run(i, source, Ty::Option(Box::new(Ty::I64)))
        .await
        .option_payload()
        .map(|n| n.as_int())
}

#[tokio::test]
async fn each_argument_is_lent_at_its_settled_type_and_none_at_another() {
    let i = Interner::new();
    run(&i, r#"inspect(41, "ab".to_string(), point(3, 4))"#, Ty::I64).await;
    let seen = SEEN.lock().expect("no test panicked holding the lock").pop().expect("inspect ran");
    assert_eq!(
        seen,
        Seen {
            n: Some(41),
            s: Some("ab".to_owned()),
            p: Some(Xy { x: 3, y: 4 }),
            n_as_string: None,
            p_as_i64: None,
            past_the_end: None,
            n_after_the_write: Some(42),
            s_after_the_write: Some("ab!".to_owned()),
        }
    );
}

#[tokio::test]
async fn a_write_to_a_by_value_argument_stays_in_the_call() {
    let i = Interner::new();
    let n = run(&i, "let n = 41;\nlet bumped = bump_one(n);\nn", Ty::I64).await;
    assert_eq!(n.as_int(), 41);
}

#[tokio::test]
async fn len_counts_the_members() {
    let i = Interner::new();
    let three = run(&i, r#"count_three(1, "a".to_string(), true)"#, Ty::I64).await;
    assert_eq!(three.as_int(), 3);
    let one = run(&i, "count_one(7)", Ty::I64).await;
    assert_eq!(one.as_int(), 1);
}

#[tokio::test]
async fn a_view_of_one_member_reads_it() {
    let i = Interner::new();
    assert_eq!(run_optional(&i, "read_one(7)").await, Some(7));
    assert_eq!(run_optional(&i, "read_one(true)").await, None);
}

#[tokio::test]
async fn a_view_moves_into_an_awaited_and_an_offloaded_call() {
    let i = Interner::new();
    assert_eq!(run_optional(&i, "read_one_awaited(9)").await, Some(9));
    assert_eq!(run_optional(&i, "read_one_offloaded(11)").await, Some(11));
}

#[tokio::test]
async fn a_reference_argument_is_lent_as_its_target() {
    let i = Interner::new();
    assert_eq!(run_optional(&i, "let n = 5;\nread_one(&n)").await, Some(5));
    assert_eq!(run_optional(&i, "let n = 5;\nbump_one(&n)").await, None);
    let bumped = run(&i, "let n = 5;\nlet bumped = bump_one(&mut n);\nn", Ty::I64).await;
    assert_eq!(bumped.as_int(), 6);
}
