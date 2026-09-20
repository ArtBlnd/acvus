//! A `&[T]` parameter reaches the Rust body as the container the script
//! lent (RFC-0047 rule 6), at the contract: the number the script returns.

use acvus_extern::{Mut, OneValue, Registry, Runtime, Shared, Slice, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

fn element_of<Rt>(rt: &Rt, view: &acvus_extern::Elements<Rt>, at: usize) -> i64
where
    Rt: Runtime,
{
    // SAFETY: `at` is below the length the view reports, the container the
    // caller lent is live for the call (RFC-0018), and every element of a
    // language `Vec<i64>` was erased from `i64` (RFC-0047 rule 1).
    unsafe { *rt.value_as_ref::<i64>(view.at(at)) }
}

#[extern_fn(effect = pure)]
fn dot<Rt>(rt: &Rt, a: Slice<i64, Shared, Rt>, b: Slice<i64, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    let (a, b) = (a.into_elements(), b.into_elements());
    assert_eq!(a.len(), b.len(), "dot takes two views of one length");
    (0..a.len())
        .map(|at| element_of(rt, &a, at) * element_of(rt, &b, at))
        .sum()
}

#[extern_fn(effect = opaque)]
fn add_into<Rt>(rt: &Rt, dst: Slice<i64, Mut, Rt>, src: Slice<i64, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    let (dst, src) = (dst.into_elements(), src.into_elements());
    assert_eq!(
        dst.len(),
        src.len(),
        "add_into takes two views of one length"
    );
    for at in 0..dst.len() {
        let sum = element_of(rt, &dst, at) + element_of(rt, &src, at);
        // SAFETY: `at` is below the length, and a `Mut` slice is an exclusive
        // take of its container, so no other name of the element is live
        // (RFC-0047 §2).
        unsafe { *dst.at_mut(at) = OneValue::<_>::erase(sum, rt) };
    }
    i64::try_from(dst.len()).expect("a view's length is an i64")
}

/// The sum of a view.
#[extern_fn(effect = pure)]
fn total<Rt>(rt: &Rt, a: Slice<i64, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    let a = a.into_elements();
    (0..a.len()).map(|at| element_of(rt, &a, at)).sum()
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
