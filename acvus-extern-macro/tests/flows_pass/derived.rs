//! The flows `#[extern_fn]` reads off a Rust signature (RFC-0079 rule 6),
//! asserted on the declaration it generates.
use acvus_extern::{
    Alignment, Closure, ClosureFn, Cross, Ctx, ExternFn, Flow, FlowEnd, Flows, Interner, OneValue,
    PassedByValue, PolyTy, Runtime, TypesOnly, Var, extern_fn, kind,
};

#[extern_fn(effect = pure)]
fn elided(s: &str) -> &str {
    s
}

#[extern_fn(effect = pure)]
fn named<'a>(s: &'a str, pat: &str) -> &'a str {
    s.strip_prefix(pat).unwrap_or(s)
}

#[extern_fn(effect = pure)]
fn shared<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() >= b.len() { a } else { b }
}

#[extern_fn(effect = pure)]
fn outlived<'a, 'b>(a: &'a str, b: &'b str) -> &'b str
where
    'a: 'b,
{
    if a.len() >= b.len() { a } else { b }
}

#[extern_fn(effect = pure)]
fn fixed(s: &str) -> &'static str {
    if s.is_empty() { "" } else { "x" }
}

#[extern_fn(effect = pure)]
fn pick<T>(a: Option<T>, b: T) -> T
where
    T: Var<kind::Type>,
{
    a.unwrap_or(b)
}

#[extern_fn(effect = pure)]
fn keep<T>(c: &mut Vec<T>, item: T)
where
    T: Var<kind::Type>,
{
    c.push(item)
}

#[extern_fn(effect = pure)]
fn unrelated<T, U>(t: T, u: U) -> Option<T>
where
    T: Var<kind::Type>,
    U: Var<kind::Type>,
{
    drop(u);
    Some(t)
}

#[extern_fn(effect = opaque)]
async fn apply<T, U, E, Rt>(ctx: &mut Ctx<'_, Rt>, t: T, f: Closure<'_, (T,), U, E, Rt>) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    f.call(ctx, (t,)).await
}

#[extern_fn(effect = pure)]
fn free<'a, Rt>(n: u64) -> Option<Closure<'a, (), i64, acvus_extern::Pure, Rt>>
where
    Rt: Runtime,
{
    let _ = n;
    None
}

fn flows_of(declared: Vec<ExternFn<TypesOnly>>) -> Flows {
    match &declared[0].decl.ty {
        PolyTy::Fn { flows, .. } => flows.get().clone(),
        other => panic!("a declaration's type is a function type, not {other:?}"),
    }
}

fn flow(to: FlowEnd, from: usize, alignment: Alignment) -> Flow {
    Flow {
        to,
        from: FlowEnd::Param(from),
        alignment,
    }
}

fn main() {
    let i = Interner::new();
    use Alignment::{Aligned, Any};
    use FlowEnd::{Param, Result};
    // An elided result takes the one lifetime its parameter names.
    assert_eq!(
        flows_of(__extern_fn_elided::<TypesOnly>(&i, None)),
        Flows::of([flow(Result, 0, Aligned)])
    );
    // A named lifetime: the result borrows `s` and not `pat`.
    assert_eq!(
        flows_of(__extern_fn_named::<TypesOnly>(&i, None)),
        Flows::of([flow(Result, 0, Aligned)])
    );
    // One lifetime on two parameters: the result may borrow either.
    assert_eq!(
        flows_of(__extern_fn_shared::<TypesOnly>(&i, None)),
        Flows::of([flow(Result, 0, Aligned), flow(Result, 1, Aligned)])
    );
    // `'a: 'b` lets a `'a` borrow stand where `'b` is named.
    assert_eq!(
        flows_of(__extern_fn_outlived::<TypesOnly>(&i, None)),
        Flows::of([flow(Result, 0, Aligned), flow(Result, 1, Aligned)])
    );
    // `'static` names no loan.
    assert_eq!(flows_of(__extern_fn_fixed::<TypesOnly>(&i, None)), Flows::none());
    // A type variable: `T`'s positions of either input reach `T`'s of the
    // result, one shape to one shape.
    assert_eq!(
        flows_of(__extern_fn_pick::<TypesOnly>(&i, None)),
        Flows::of([flow(Result, 0, Aligned), flow(Result, 1, Aligned)])
    );
    // A write through `&mut Vec<T>` takes every `T` handed in.
    assert_eq!(
        flows_of(__extern_fn_keep::<TypesOnly>(&i, None)),
        Flows::of([flow(Param(0), 0, Any), flow(Param(0), 1, Any)])
    );
    // Two type variables do not meet.
    assert_eq!(
        flows_of(__extern_fn_unrelated::<TypesOnly>(&i, None)),
        Flows::of([flow(Result, 0, Aligned)])
    );
    // A closure maps what it is handed to what it returns: `U` takes from
    // `t` through it, and from its captures; its captures take `t`.
    assert_eq!(
        flows_of(__extern_fn_apply::<TypesOnly>(&i, None)),
        Flows::of([
            flow(Result, 0, Any),
            flow(Result, 1, Any),
            flow(Param(1), 0, Any),
            flow(Param(1), 1, Any),
        ])
    );
    // A lifetime no parameter names is free: the handler is generic over
    // it, so what the result holds there names no loan.
    assert_eq!(flows_of(__extern_fn_free::<TypesOnly>(&i, None)), Flows::none());
}
