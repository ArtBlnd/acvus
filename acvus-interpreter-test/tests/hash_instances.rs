//! A uniform value lent to a `#` parameter inside a running script
//! (scenarios.md S12). Ignored until the compiler side exists; see the
//! test's own doc for where.

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

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [norm],
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

/// A uniform `Vec<f64>` lent to a `#` parameter is materialized in its
/// storage before the call and erased back after it (scenarios.md S12).
/// Two compiler steps are not built, in this order. First, `norm`'s
/// generic signature carries a uniform slot, not a `ρ`, so the checker
/// reports `no instance of the signature has the call type Fn(&Vec<Float>)
/// -> Float` before any conversion is considered; the `ρ` is deferred by
/// both.md. Second, `check_args` in `acvus-mir/src/typeck.rs` registers a
/// conversion on the argument expression, and a cast settled there is
/// lowered as a value conversion of the reference itself, not of the place
/// it names. Expected once both exist: `5.0`, with the storage of `x`
/// uniform again afterwards.
#[tokio::test]
#[ignore = "S12: the generic signature has no `ρ`, and the place rule for a lent argument is compiler work"]
async fn a_uniform_vec_lent_to_a_specialized_parameter_is_converted_in_place() {
    let v = run("let x = vec([3.0, 4.0]); norm(&x)").await;
    assert_eq!(v.as_float(), 5.0);
}
