//! A uniform value lent to a `#` parameter inside a running script
//! (scenarios.md S12, reference-cast-in-place.md): the argument's storage
//! is cast into the callee's representation before the call and back
//! after it, so the callee reads and writes the caller's storage.

use acvus_extern::{Monomorphize, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_utils::Interner;

trait Float: Copy + Send + Sync + 'static {
    const ZERO: Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
}

impl Float for f64 {
    const ZERO: Self = 0.0;

    fn mul_add(self, a: Self, b: Self) -> Self {
        self + a * b
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

#[extern_fn(effect = pure)]
fn norm<T>(v: &Vec<T>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    v.iter().fold(T::ZERO, |acc, x| acc.mul_add(*x, *x)).sqrt()
}

#[extern_fn(effect = pure)]
fn scale<T>(v: &mut Vec<T>, k: T)
where
    T: Monomorphize<(f64,)> + Float,
{
    for x in v.iter_mut() {
        *x = T::ZERO.mul_add(*x, k);
    }
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [norm, scale],
    }
}

async fn run(source: &str) -> Value {
    let i = Interner::new();
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(registry());
    run_script_mode_with_externs(&i, source, Context::default(), registries)
        .await
        .value
}

/// `vec([..])` runs the generic `std::vec`, so `x` holds a uniform
/// `Vec<Float>`; `norm` exists only at `#f64` and takes `&Vec<#f64>`.
#[tokio::test]
async fn a_uniform_vec_lent_to_a_specialized_parameter_is_converted_in_place() {
    let v = run("let x = vec([3.0, 4.0]); norm(&x)").await;
    assert_eq!(v.as_float(), 5.0);
}

#[tokio::test]
async fn a_callee_s_writes_through_a_mutable_lend_survive_the_cast_back() {
    let v = run("let x = vec([3.0, 4.0]); scale(&mut x, 2.0); norm(&x)").await;
    assert_eq!(v.as_float(), 10.0);
}
