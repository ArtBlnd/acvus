//! `E: Suspends` on an `#[extern_fn]` is the declaration's effect bound at
//! `E`'s index, and an effect-generic `async fn` at such a variable needs no
//! `sync =` twin (RFC-0011 rule 5, RFC-0046 rule 5).

use acvus_extern::{
    Closure, ClosureFn, Ctx, EffectTerm, EffectVarBound, Externs, FnKind, Interner, PolyTy,
    QualifiedRef, Registry, Runtime, Suspends, Task, TypesOnly, Var, extern_fn, extern_registry,
    extern_signature, kind,
};

#[extern_fn(effect = pure)]
async fn both<D, E, F, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    first: Closure<'_, (i64,), i64, D, Rt>,
    second: Closure<'_, (i64,), i64, E, Rt>,
    third: Closure<'_, (i64,), i64, F, Rt>,
) -> i64
where
    D: Var<kind::Effect>,
    E: Var<kind::Effect> + Suspends,
    F: Var<kind::Effect>,
    Rt: Runtime,
{
    first.call(ctx, (1,)).await + second.call(ctx, (2,)).await + third.call(ctx, (3,)).await
}

#[extern_fn(effect = E)]
async fn each<E, Rt>(ctx: &mut Ctx<'_, Rt>, f: Closure<'_, (i64,), i64, E, Rt>) -> i64
where
    E: Var<kind::Effect> + Suspends,
    Rt: Runtime,
{
    f.call(ctx, (1,)).await + f.call(ctx, (2,)).await
}

extern_signature! {
    ns: "t",
    effect = E,
    fn run<F, E, Rt>(f: F) -> i64
    where
        F: Var<kind::Type>,
        E: Var<kind::Effect>,
        Rt: Runtime;
}

#[extern_fn(instance_of = run, effect = E)]
async fn run_each<E, Rt>(ctx: &mut Ctx<'_, Rt>, f: Closure<'_, (i64,), i64, E, Rt>) -> i64
where
    E: Var<kind::Effect> + Suspends,
    Rt: Runtime,
{
    f.call(ctx, (1,)).await
}

fn registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "t",
        signatures: [run],
        fns: [each, run_each],
    }
}

#[test]
fn a_suspending_effect_variable_is_bounded_at_its_index() {
    let i = Interner::new();
    let declared = __extern_fn_both::<TypesOnly>(&i, None);
    let [both] = declared.as_slice() else {
        panic!("one declaration, got {}", declared.len());
    };
    assert_eq!(
        both.decl.effect_bounds,
        vec![
            EffectVarBound::Any,
            EffectVarBound::Suspends,
            EffectVarBound::Any,
        ]
    );
}

#[test]
fn an_async_fn_at_a_suspending_effect_needs_no_twin() {
    let i = Interner::new();
    let reg = Externs::combine(vec![registry::<TypesOnly>()], &i).expect("registries combine");
    let each = QualifiedRef::qualified(i.intern("t"), i.intern("each"));
    let declared = reg
        .functions
        .iter()
        .find(|f| f.qref == each)
        .expect("t::each is declared");
    let PolyTy::Fn { effect, .. } = &declared.ty else {
        panic!("not a function type: {:?}", declared.ty);
    };
    assert_eq!(effect, &EffectTerm::Var(0));
    let FnKind::Extern {
        effect_bounds,
        instances,
        ..
    } = &declared.kind
    else {
        panic!("t::each is extern");
    };
    assert_eq!(effect_bounds, &vec![EffectVarBound::Suspends]);
    assert!(instances.concrete.is_empty(), "{instances:?}");
    assert!(instances.generic.is_some(), "{instances:?}");
    let tasks: Vec<Task> = reg.handlers[&each].iter().map(|h| h.task()).collect();
    assert_eq!(tasks, vec![Task::Async]);
}

#[test]
fn a_signature_s_instance_carries_its_suspending_bound() {
    let i = Interner::new();
    let reg = Externs::combine(vec![registry::<TypesOnly>()], &i).expect("registries combine");
    let signature = QualifiedRef::qualified(i.intern("t"), i.intern("run"));
    let declared = reg
        .functions
        .iter()
        .find(|f| f.qref == signature)
        .expect("t::run is declared");
    let FnKind::Extern {
        effect_bounds,
        instances,
        ..
    } = &declared.kind
    else {
        panic!("t::run is extern");
    };
    assert!(effect_bounds.is_empty(), "{effect_bounds:?}");
    let bounds: Vec<&Vec<EffectVarBound>> = instances
        .concrete
        .iter()
        .map(|sig| &sig.effect_bounds)
        .collect();
    assert_eq!(bounds, vec![&vec![EffectVarBound::Suspends]]);
}
