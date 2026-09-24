//! An erased closure is `fn(A)` turned into `fn(Value)`: every later call of
//! it downcasts. `OneValue::erase` takes the glue's `Crossing`, which a
//! handler is never handed, so a handler cannot erase the closure it was
//! called with (acvus-extern's rule, RFC-0068 rule 1).
use acvus_extern::{Closure, Ctx, OneValue, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = opaque)]
fn forget<E, Rt>(ctx: &mut Ctx<'_, Rt>, f: Closure<'_, (i64,), i64, E, Rt>) -> i64
where
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let _word = <Closure<'static, (i64,), i64, E, Rt> as OneValue<Rt>>::erase(f, ctx.rt);
    0
}

fn main() {}
