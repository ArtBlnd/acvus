//! A `&[T]` parameter reaches the Rust body as the container the script
//! lent (RFC-0047 rule 6), at the contract: the number the script returns.

use acvus_extern::{Erased, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn dot<Rt>(a: &[Erased<Rt, i64>], b: &[Erased<Rt, i64>]) -> i64
where
    Rt: Runtime,
{
    assert_eq!(a.len(), b.len(), "dot takes two views of one length");
    a.iter().zip(b).map(|(x, y)| **x * **y).sum()
}

#[extern_fn(effect = opaque)]
fn add_into<Rt>(dst: &mut [Erased<Rt, i64>], src: &[Erased<Rt, i64>]) -> i64
where
    Rt: Runtime,
{
    assert_eq!(
        dst.len(),
        src.len(),
        "add_into takes two views of one length"
    );
    for (d, s) in dst.iter_mut().zip(src) {
        **d += **s;
    }
    i64::try_from(dst.len()).expect("a view's length is an i64")
}

#[extern_fn(effect = pure)]
fn total<Rt>(a: &[Erased<Rt, i64>]) -> i64
where
    Rt: Runtime,
{
    a.iter().map(|x| **x).sum()
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        fns: [dot, add_into, total],
    });
    regs
}

async fn answer(source: &str) -> i64 {
    let i = Interner::new();
    run_script_with_externs(&i, source, Context::default(), registries(), Ty::I64)
        .await
        .value
        .as_int()
}

#[tokio::test]
async fn a_borrowed_container_reaches_a_slice_parameter() {
    let source = "let a = vec([1, 2, 3]); let b = vec([10, 20, 30]); dot(&a, &b)";
    assert_eq!(answer(source).await, 140);
}

#[tokio::test]
async fn a_borrowed_element_reaches_a_slice_parameter() {
    let source = "\
let rows = vec([vec([1, 2, 3]), vec([4, 5, 6])]); \
let b = vec([10, 20, 30]); \
let z = len(&b) - len(&b); \
let one = z + 1u64; \
dot(&rows[one], &b)";
    assert_eq!(answer(source).await, 320);
}

#[tokio::test]
async fn an_exclusive_slice_parameter_writes_into_the_container() {
    let source = "\
let a = vec([1, 2, 3]); \
let b = vec([10, 20, 30]); \
let n = add_into(&mut a, &b); \
let z = len(&a) - len(&a); \
a[z] * n";
    assert_eq!(answer(source).await, 33);
}

/// RFC-0047 rule 6 with RFC-0059 rule 7.
#[tokio::test]
async fn a_single_slice_parameter_takes_the_window() {
    assert_eq!(answer("let v = vec([1, 2, 3, 4]); total(&v)").await, 10);
}
