//! RFC-0067 at the contract that decides the design: an extension type
//! whose payload is the stage tuple its element-type list names, crossed the
//! boundary and read back at each length, with the closure called typed.
//!
//! `Pipe` is `Iter` with two stage kinds. What `signature_effect.rs` could not
//! ask, because its payload was an `i64` shared by every instantiation, is
//! whether the glue hands a per-length instance the stage tuple its own `Ts`
//! spells, with `Owned` at the leaves and the nesting intact.

use std::marker::PhantomData;

use acvus_extern::Ctx;
use acvus_extern::{
    Bottom, Chosen, ChosenNth, Closure, ClosureFn, Cross, Erased, ExternType, OneValue,
    PassedByValue, Registry, Runtime, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// A type-equality witness, the identity as a value. Only `Same::<T, T>::new`
/// mints one, so a stage carrying `Same<In, Out>` has `In` equal to `Out`
/// without the pipeline asking for specialization.
pub struct Same<A, B>(fn(A) -> B);

// SAFETY: the witness holds the identity function and no value of `A` or
// `B`.
unsafe impl<A, B> acvus_extern::Branded for Same<A, B>
where
    A: 'static,
    B: 'static,
{
    type At<'a> = Self;
}

impl<T> Same<T, T> {
    pub fn new() -> Self {
        Same(|x| x)
    }
}

impl<A, B> Same<A, B> {
    pub fn apply(&self, a: A) -> B {
        (self.0)(a)
    }
}

#[derive(acvus_extern::Branded)]
pub enum Stage<'a, In, Out, E, Rt>
where
    In: Var<kind::Type>,
    Out: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Map(Closure<'a, (In,), Out, E, Rt>),
    Take { remaining: u64, same: Same<In, Out> },
}

#[derive(acvus_extern::Branded)]
pub struct Source<T, Rt>
where
    Rt: Runtime,
{
    items: std::vec::IntoIter<T>,
    runtime: PhantomData<Rt>,
}

impl<T, Rt> Source<T, Rt>
where
    Rt: Runtime,
{
    fn pull(&mut self) -> Option<T> {
        self.items.next()
    }
}

/// One stage and the pipeline beneath it.
#[derive(acvus_extern::Branded)]
pub struct Stages<S, Rest> {
    stage: S,
    rest: Rest,
}

/// The list of element types a pipeline's elements passed through, as nested
/// pairs with `()` the empty list. `Body` is the stage stack the list names.
pub trait TypeList<Rt>: Send + Sync + 'static
where
    Rt: Runtime,
{
    type Body<'a, O, E>: Send + Sync + 'a
    where
        O: Var<kind::Type>,
        E: Var<kind::Effect>;

    fn pull<O, E>(body: &mut Self::Body<'_, O, E>, ctx: &mut Ctx<'_, Rt>) -> Option<O>
    where
        O: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
        E: Var<kind::Effect>;
}

impl<Rt> TypeList<Rt> for ()
where
    Rt: Runtime,
{
    type Body<'a, O, E>
        = Source<O, Rt>
    where
        O: Var<kind::Type>,
        E: Var<kind::Effect>;

    fn pull<O, E>(body: &mut Source<O, Rt>, _: &mut Ctx<'_, Rt>) -> Option<O>
    where
        O: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
        E: Var<kind::Effect>,
    {
        body.pull()
    }
}

impl<T, Ts, Rt> TypeList<Rt> for (T, Ts)
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
    Ts: TypeList<Rt>,
    Rt: Runtime,
{
    type Body<'a, O, E>
        = Stages<Stage<'a, T, O, E, Rt>, Ts::Body<'a, T, E>>
    where
        O: Var<kind::Type>,
        E: Var<kind::Effect>;

    fn pull<O, E>(body: &mut Self::Body<'_, O, E>, ctx: &mut Ctx<'_, Rt>) -> Option<O>
    where
        O: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
        E: Var<kind::Effect>,
    {
        let Stages { stage, rest } = body;
        match stage {
            Stage::Map(f) => {
                let item = Ts::pull::<T, E>(rest, ctx)?;
                Some(f.call_now(ctx, (item,)))
            }
            Stage::Take { remaining, same } => {
                *remaining = remaining.checked_sub(1)?;
                Some(same.apply(Ts::pull::<T, E>(rest, ctx)?))
            }
        }
    }
}

/// The stand-in a shared signature's own type carries: a `Ts` bounded by
/// `Chosen` is `ChosenNth<N>` while the declaration's type is built, and
/// `Bottom` says no pipeline of that type is a value.
impl<const N: usize, Rt> TypeList<Rt> for ChosenNth<N>
where
    Rt: Runtime,
{
    type Body<'a, O, E>
        = Bottom
    where
        O: Var<kind::Type>,
        E: Var<kind::Effect>;

    fn pull<O, E>(body: &mut Bottom, _: &mut Ctx<'_, Rt>) -> Option<O>
    where
        O: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
        E: Var<kind::Effect>,
    {
        match *body {}
    }
}

#[derive(ExternType)]
#[extern_type(name = "Pipe")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Pipe<'a, Ts, O, E, I, Rt>(<Ts as TypeList<Rt>>::Body<'a, O, E>, PhantomData<I>)
where
    Ts: Var<kind::Type> + TypeList<Rt> + Chosen,
    O: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime;

impl<'a, Ts, O, E, I, Rt> Pipe<'a, Ts, O, E, I, Rt>
where
    Ts: Var<kind::Type> + TypeList<Rt>,
    O: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    fn push<U>(self, stage: Stage<'a, O, U, E, Rt>) -> Pipe<'a, (O, Ts), U, E, I, Rt>
    where
        U: Var<kind::Type>,
    {
        Pipe(
            Stages {
                stage,
                rest: self.0,
            },
            PhantomData,
        )
    }

    fn drain(mut self, ctx: &mut Ctx<'_, Rt>) -> Vec<O> {
        let mut out = Vec::new();
        while let Some(item) = Ts::pull::<O, E>(&mut self.0, ctx) {
            out.push(item);
        }
        out
    }
}

#[extern_fn(effect = pure)]
fn ints<'a, T, E, I, Rt>(items: Vec<T>) -> Pipe<'a, (), T, E, I, Rt>
where
    T: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Pipe(
        Source {
            items: items.into_iter(),
            runtime: PhantomData,
        },
        PhantomData,
    )
}

mod sig {
    use acvus_extern::{Closure, extern_signature};

    use super::Pipe;

    extern_signature! {
        ns: "p",
        fn step<Ts, T, U, E, I, Rt>(
            it: Pipe<Ts, T, E, I, Rt>,
            f: Closure<'_, (T,), U, E, Rt>,
        ) -> Pipe<(T, Ts), U, E, I, Rt>
        where
            Ts: Var<kind::Type> + Chosen,
            T: Var<kind::Type>,
            U: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "p",
        fn cut<Ts, T, E, I, Rt>(it: Pipe<Ts, T, E, I, Rt>, n: u64) -> Pipe<(T, Ts), T, E, I, Rt>
        where
            Ts: Var<kind::Type> + Chosen,
            T: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "p",
        effect = E,
        fn total<Ts, O, E, I, Rt>(it: Pipe<Ts, O, E, I, Rt>) -> i64
        where
            Ts: Var<kind::Type> + Chosen,
            O: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime;
    }
}

macro_rules! adaptor_instances {
    ($step:ident, $cut:ident, [$($v:ident),*], $($ts:tt)+) => {
        #[extern_fn(instance_of = sig::step, effect = pure)]
        fn $step<'a, $($v,)* T, U, E, I, Rt>(
            it: Pipe<'a, $($ts)+, T, E, I, Rt>,
            f: Closure<'a, (T,), U, E, Rt>,
        ) -> Pipe<'a, (T, $($ts)+), U, E, I, Rt>
        where
            $($v: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,)*
            T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
            U: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            it.push(Stage::Map(f))
        }

        #[extern_fn(instance_of = sig::cut, effect = pure)]
        fn $cut<'a, $($v,)* T, E, I, Rt>(
            it: Pipe<'a, $($ts)+, T, E, I, Rt>,
            n: u64,
        ) -> Pipe<'a, (T, $($ts)+), T, E, I, Rt>
        where
            $($v: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,)*
            T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            it.push(Stage::Take { remaining: n, same: Same::new() })
        }
    };
}

adaptor_instances!(step_0, cut_0, [], ());
adaptor_instances!(step_1, cut_1, [A], (A, ()));
adaptor_instances!(step_2, cut_2, [A, B], (A, (B, ())));

macro_rules! total_instance {
    ($name:ident, $now:ident, [$($v:ident),*], $($ts:tt)+) => {
        fn $now<$($v,)* O, E, I, Rt>(
            ctx: &mut Ctx<'_, Rt>,
            it: Pipe<'_, $($ts)+, O, E, I, Rt>,
        ) -> i64
        where
            $($v: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,)*
            O: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            let rt = ctx.rt;
            it.drain(ctx)
                .into_iter()
                // SAFETY: the pipe's element type is the `i64` the checker
                // matched at this handler's parameter.
                .map(|x| *unsafe { Erased::<Rt, i64>::from_value_of(rt, x.erase(rt)) }.as_ref(rt))
                .sum()
        }

        #[extern_fn(instance_of = sig::total, effect = E, sync = $now)]
        async fn $name<$($v,)* O, E, I, Rt>(
            ctx: &mut Ctx<'_, Rt>,
            it: Pipe<'_, $($ts)+, O, E, I, Rt>,
        ) -> i64
        where
            $($v: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,)*
            O: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + PassedByValue<Rt>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            let rt = ctx.rt;
            it.drain(ctx)
                .into_iter()
                // SAFETY: the pipe's element type is the `i64` the checker
                // matched at this handler's parameter.
                .map(|x| *unsafe { Erased::<Rt, i64>::from_value_of(rt, x.erase(rt)) }.as_ref(rt))
                .sum()
        }
    };
}

total_instance!(total_0, total_0_now, [], ());
total_instance!(total_1, total_1_now, [A], (A, ()));
total_instance!(total_2, total_2_now, [A, B], (A, (B, ())));
total_instance!(total_3, total_3_now, [A, B, C], (A, (B, (C, ()))));

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "p",
        types: [Pipe<_, _, _, _, AcvusRuntime>],
        signatures: [sig::step, sig::cut, sig::total],
        fns: [
            ints,
            step_0, step_1, step_2,
            cut_0, cut_1, cut_2,
            total_0, total_1, total_2, total_3,
        ],
    }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut rs = acvus_ext::std_registries::<AcvusRuntime>();
    rs.push(registry());
    rs
}

async fn run_i64(source: &str) -> i64 {
    let i = Interner::new();
    run_script_mode_with_externs(&i, source, Context::default(), registries(), Ty::I64)
        .await
        .value
        .as_int()
}

#[tokio::test]
async fn a_pipeline_whose_payload_is_its_stage_tuple_crosses_at_every_length() {
    assert_eq!(run_i64("ints(vec([1, 2, 3])) | total()").await, 6);
    assert_eq!(
        run_i64("ints(vec([1, 2, 3])) | step(|x| -> x * 10) | total()").await,
        60
    );
    assert_eq!(
        run_i64("ints(vec([1, 2, 3])) | step(|x| -> x * 10) | cut(2) | total()").await,
        30
    );
    assert_eq!(
        run_i64(
            "ints(vec([1, 2, 3])) | step(|x| -> x * 10) | cut(2) | step(|x| -> x + 1) | total()"
        )
        .await,
        32
    );
}
