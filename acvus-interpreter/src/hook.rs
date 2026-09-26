//! A hook (RFC-0101): a dynamic extern whose body is a closure the host
//! binds, handed the call's `Args` view and `Output` and nothing else.
//!
//! The view, the output and the glue that builds them are acvus-extern's,
//! unchanged: a hook is this interpreter's feature, and the extern contract
//! every runtime shares gains nothing for it.

use std::sync::{Arc, OnceLock};

use acvus_extern::{
    Args, AsyncFactory, ByArgs, ByOutput, Contribution, Crossing, Ctx, ExternFn, ExternHandler, Finished, FnDecl,
    Gives, Instances, Laws, Manifest, Nth, Output, Owned, PolyVars, Reaches, Registry, Returns, Runtime, TyArg,
    TyVarBound, async_glue, kind,
};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{Effect, EffectTerm, Flows, ParamTerm, Poly, PolyTy, Task};
use acvus_utils::Interner;
use futures::future::BoxFuture;

use crate::host::{Access, Cause, HostError, Named, Program, Refusal};
use crate::port::end_run;
use crate::runtime::AcvusRuntime;
use crate::value::Value;

/// One argument position, and the result's payload: the declaration's own
/// type variable, settled at each call site. The glue fills it with a word
/// no body opens (RFC-0023 rule 10), and the host never names it.
type SiteTyped = Owned<AcvusRuntime>;

/// What a closure bound to a hook of `N` arguments is handed for them:
/// `()` for none, else the `Args` view of RFC-0097 rule 1 over `N`
/// positions.
pub type HookArgs<'c, const N: usize> = <HookArity<N> as sealed::Sealed>::View<'c>;

/// A hook call's result, filled at the payload type its site settled
/// (RFC-0097 rule 3).
pub type HookOutput<'c> = Output<'c, SiteTyped, AcvusRuntime>;

/// What `HookOutput::finish` seals: the call's `Some`, or its `None`.
pub type HookFinished<'c> = Finished<'c, SiteTyped, AcvusRuntime>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookEffect {
    Pure,
    Idempotent,
    Opaque,
}

impl HookEffect {
    /// The declared effect at `Async`: the closure may wait, and the call
    /// suspends its caller until the closure's future ends.
    fn effect(self) -> Effect {
        let declared = match self {
            HookEffect::Pure => Effect::PURE,
            HookEffect::Idempotent => Effect::IDEMPOTENT,
            HookEffect::Opaque => Effect::OPAQUE,
        };
        declared.at_task(Task::Async)
    }
}

/// What `Program::bind` refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookPart {
    /// The closure takes another number of arguments than the hook declares.
    Arity,
    /// The hook is bound already.
    Bound,
}

// -- The closure's arguments ---------------------------------------------

/// The arity a binding is written for, `Program::bind::<N, _>`.
pub struct HookArity<const N: usize>;

mod sealed {
    use super::{Binding, Slot};

    pub trait Sealed: Sized + 'static {
        type View<'c>: Send;

        fn binding(slot: &Slot) -> Option<&Binding<Self>>;
    }
}

/// The arities a hook takes: `0` through `acvus_extern::MOST_MEMBERS`.
pub trait HookParams: sealed::Sealed {}

type Body<P> = dyn for<'c> Fn(<P as sealed::Sealed>::View<'c>, HookOutput<'c>) -> BoxFuture<'c, HookFinished<'c>>
    + Send
    + Sync;

pub struct Binding<P>
where
    P: sealed::Sealed,
{
    hook: String,
    body: OnceLock<Box<Body<P>>>,
}

macro_rules! arg {
    ($at:literal) => {
        SiteTyped
    };
}

macro_rules! slot {
    ($($variant:ident: $arity:literal ($($at:literal),*);)+) => {
        pub enum Slot {
            $($variant(Arc<Binding<HookArity<$arity>>>),)+
        }

        impl Slot {
            fn of(hook: &str, arity: usize) -> Option<Slot> {
                match arity {
                    $($arity => Some(Slot::$variant(Arc::new(Binding::new(hook)))),)+
                    _ => None,
                }
            }

            fn shared(&self) -> Slot {
                match self {
                    $(Slot::$variant(binding) => Slot::$variant(Arc::clone(binding)),)+
                }
            }

            fn arity(&self) -> usize {
                match self {
                    $(Slot::$variant(_) => $arity,)+
                }
            }

            fn is_bound(&self) -> bool {
                match self {
                    $(Slot::$variant(binding) => binding.body.get().is_some(),)+
                }
            }
        }

        $(
            impl sealed::Sealed for HookArity<$arity> {
                type View<'c> = slot!(@view $($at),*);

                fn binding(slot: &Slot) -> Option<&Binding<Self>> {
                    match slot {
                        Slot::$variant(binding) => Some(binding),
                        _ => None,
                    }
                }
            }

            impl HookParams for HookArity<$arity> {}
        )+
    };
    (@view) => { () };
    (@view $($at:literal),+) => { Args<'c, ($(arg!($at),)+), AcvusRuntime> };
}

slot! {
    Zero: 0 ();
    One: 1 (0);
    Two: 2 (0, 1);
    Three: 3 (0, 1, 2);
    Four: 4 (0, 1, 2, 3);
    Five: 5 (0, 1, 2, 3, 4);
    Six: 6 (0, 1, 2, 3, 4, 5);
    Seven: 7 (0, 1, 2, 3, 4, 5, 6);
    Eight: 8 (0, 1, 2, 3, 4, 5, 6, 7);
}

impl<P> Binding<P>
where
    P: sealed::Sealed,
{
    fn new(hook: &str) -> Self {
        Binding {
            hook: hook.to_owned(),
            body: OnceLock::new(),
        }
    }

    /// A run a host starts refuses a program with an unbound hook before it
    /// starts (`Compiled::run_over`). A page's load runs a key's init
    /// outside any run, so an init calling an unbound hook ends here, with
    /// the same `Unbound`.
    fn called<'c>(&self, ctx: &'c mut Ctx<'_, AcvusRuntime>, args: P::View<'c>, out: HookOutput<'c>) -> BoxFuture<'c, Value> {
        let Some(body) = self.body.get() else {
            end_run(HostError::Unbound {
                hook: self.hook.clone(),
            })
        };
        let pending = body(args, out);
        Box::pin(async move {
            let finished = pending.await;

            // `give` writes the payload where it gives `true`, and only then
            // is the slot read.
            let mut payload = [Value::unit()];
            // SAFETY: this is the runtime, crossing the one value the call's
            // own `Output` built at the payload type its site settled.
            let rt = unsafe { Crossing::new(ctx.rt) };
            match finished.give(rt, &mut payload) {
                true => ctx.rt.some(payload[0]),
                false => ctx.rt.none(),
            }
        })
    }
}

// -- Declaring a hook ----------------------------------------------------

pub(crate) struct HookDecl {
    name: String,
    effect: HookEffect,
    slot: Slot,
}

impl HookDecl {
    /// `None` past the `acvus_extern::MOST_MEMBERS` arguments one `Args`
    /// view holds.
    pub(crate) fn new(name: &str, arity: usize, effect: HookEffect) -> Option<HookDecl> {
        Some(HookDecl {
            name: name.to_owned(),
            effect,
            slot: Slot::of(name, arity)?,
        })
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn registry(&self) -> Registry<AcvusRuntime> {
        let name = self.name.clone();
        let effect = self.effect.effect();
        let slot = self.slot.shared();
        Registry::new(move |interner: &Interner| {
            let mut contribution = Contribution::of(Manifest {
                types: Vec::new(),
                signatures: Vec::new(),
                fns: Vec::new(),
            });
            let declaring = Declaring {
                interner,
                name: &name,
                effect: effect.clone(),
            };
            contribution.declare(declaring.of(&slot));
            contribution
        })
    }

    pub(crate) fn compiled(self) -> CompiledHook {
        CompiledHook { slot: self.slot }
    }
}

struct Declaring<'a> {
    interner: &'a Interner,
    name: &'a str,
    effect: Effect,
}

macro_rules! declared {
    ($declaring:ident, $binding:ident; $result:literal;) => {{
        let vars = PolyVars::fresh($result + 1, 0, 0, 0);
        let result = <Option<Nth<kind::Type, $result>> as TyArg>::poly_ty($declaring.interner, &vars);
        let binding = Arc::clone($binding);
        let handler = async_glue::<AcvusRuntime, _, (ByOutput,)>(
            move |ctx: &mut Ctx<'_, AcvusRuntime>, (out,)| {
                let out: HookOutput<'_> = out.take();
                binding.called(ctx, (), out)
            },
        );
        $declaring.extern_fn(&vars, Vec::new(), result, handler)
    }};
    ($declaring:ident, $binding:ident; $result:literal; $($at:literal),+) => {{
        let vars = PolyVars::fresh($result + 1, 0, 0, 0);
        let params = vec![$(
            ParamTerm::<Poly>::new(
                $declaring.interner.intern(&format!("{}.{}", $declaring.name, $at)),
                <Nth<kind::Type, $at> as TyArg>::poly_ty($declaring.interner, &vars),
            )
        ),+];
        let result = <Option<Nth<kind::Type, $result>> as TyArg>::poly_ty($declaring.interner, &vars);
        let binding = Arc::clone($binding);
        let handler = async_glue::<AcvusRuntime, _, (ByArgs<($(arg!($at),)+)>, ByOutput)>(
            move |ctx: &mut Ctx<'_, AcvusRuntime>, (args, out)| {
                let args: Args<'_, ($(arg!($at),)+), AcvusRuntime> = args.take();
                let out: HookOutput<'_> = out.take();
                binding.called(ctx, args, out)
            },
        );
        $declaring.extern_fn(&vars, params, result, handler)
    }};
}

impl Declaring<'_> {
    fn of(&self, slot: &Slot) -> ExternFn<AcvusRuntime> {
        match slot {
            Slot::Zero(binding) => declared!(self, binding; 0;),
            Slot::One(binding) => declared!(self, binding; 1; 0),
            Slot::Two(binding) => declared!(self, binding; 2; 0, 1),
            Slot::Three(binding) => declared!(self, binding; 3; 0, 1, 2),
            Slot::Four(binding) => declared!(self, binding; 4; 0, 1, 2, 3),
            Slot::Five(binding) => declared!(self, binding; 5; 0, 1, 2, 3, 4),
            Slot::Six(binding) => declared!(self, binding; 6; 0, 1, 2, 3, 4, 5),
            Slot::Seven(binding) => declared!(self, binding; 7; 0, 1, 2, 3, 4, 5, 6),
            Slot::Eight(binding) => declared!(self, binding; 8; 0, 1, 2, 3, 4, 5, 6, 7),
        }
    }

    fn extern_fn<H>(&self, vars: &PolyVars, params: Vec<ParamTerm<Poly>>, result: PolyTy, handler: H) -> ExternFn<AcvusRuntime>
    where
        H: AsyncFactory<AcvusRuntime> + 'static,
    {
        let bounds = std::iter::repeat_n(TyVarBound::Any, params.len())
            .chain([TyVarBound::Settled])
            .collect();
        ExternFn {
            decl: FnDecl {
                qref: QualifiedRef::root(self.interner.intern(self.name)),
                ty: PolyTy::Fn {
                    params,
                    ret: Box::new(result),
                    captures: Vec::new(),
                    effect: EffectTerm::Known(self.effect.clone()),
                    flows: Flows::none().into(),
                },
                bounds,
                effect_bounds: Vec::new(),
                coercion: None,
                instance_of: None,
                requires: Vec::new(),
                names: vars.names(),
                laws: Laws::None,
                ensures: Vec::new(),
                reaches: Reaches::Lent,
                returns: Returns::Unstated,
                means: None,
                cost: None,
            },
            instances: Instances::generic(ExternHandler::awaited(handler)),
        }
    }
}

pub(crate) struct CompiledHook {
    slot: Slot,
}

impl CompiledHook {
    pub(crate) fn is_bound(&self) -> bool {
        self.slot.is_bound()
    }
}

// -- Binding -------------------------------------------------------------

impl<A> Program<A>
where
    A: Access,
{
    /// Binds `hook`, declared with `N` arguments, to `body`, which each call
    /// runs on the call's lent arguments and whose `Finished` is the call's
    /// result. `body` may capture the host's state, and is handed no context
    /// and no storage (RFC-0101 rule 3).
    ///
    /// Nothing it is lent outlives the call. A closure that keeps what
    /// `with` lends it does not compile:
    ///
    /// ```compile_fail,E0521
    /// use std::sync::{Arc, Mutex};
    /// use acvus_interpreter::{HookEffect, Host, SequentialExecutor, Source};
    ///
    /// let program = Host::new(acvus_ext::std_registries())
    ///     .hook("keep", 1, HookEffect::Opaque)
    ///     .entry::<(), i64>("main", Source::Script("match keep(\"a\".to_string()) { Some(v) => v, None => 0 }"))
    ///     .compile(SequentialExecutor)
    ///     .unwrap();
    /// let kept: Arc<Mutex<Vec<&'static String>>> = Arc::new(Mutex::new(Vec::new()));
    /// program
    ///     .bind::<1, _>("keep", move |args, out| {
    ///         let kept = Arc::clone(&kept);
    ///         Box::pin(async move {
    ///             args.with(0, |text: &String| kept.lock().unwrap().push(text));
    ///             out.finish()
    ///         })
    ///     })
    ///     .unwrap();
    /// ```
    ///
    /// nor does one that sends the view itself out of the call:
    ///
    /// ```compile_fail,E0521
    /// use std::sync::mpsc;
    /// use acvus_interpreter::{HookArgs, HookEffect, Host, SequentialExecutor, Source};
    ///
    /// let program = Host::new(acvus_ext::std_registries())
    ///     .hook("keep", 1, HookEffect::Opaque)
    ///     .entry::<(), i64>("main", Source::Script("match keep(1) { Some(v) => v, None => 0 }"))
    ///     .compile(SequentialExecutor)
    ///     .unwrap();
    /// let (send, _receive) = mpsc::sync_channel::<HookArgs<'static, 1>>(1);
    /// program
    ///     .bind::<1, _>("keep", move |args, out| {
    ///         send.send(args).unwrap();
    ///         Box::pin(async move { out.finish() })
    ///     })
    ///     .unwrap();
    /// ```
    pub fn bind<const N: usize, F>(&self, hook: &str, body: F) -> Result<(), HostError>
    where
        HookArity<N>: HookParams,
        F: for<'c> Fn(HookArgs<'c, N>, HookOutput<'c>) -> BoxFuture<'c, HookFinished<'c>> + Send + Sync + 'static,
    {
        let Some(declared) = self.compiled.hooks.get(hook) else {
            return Err(HostError::NotInGraph {
                what: Named::Hook(hook.to_owned()),
            });
        };
        let refused = |part: HookPart, message: String| {
            HostError::Refused(vec![Refusal {
                cause: Some(Cause::Hook {
                    hook: hook.to_owned(),
                    part,
                }),
                ..Refusal::of(None, message)
            }])
        };
        let Some(binding) = <HookArity<N> as sealed::Sealed>::binding(&declared.slot) else {
            let message = format!(
                "the hook `{hook}` takes {} arguments, and the closure bound to it takes {N}",
                declared.slot.arity()
            );
            return Err(refused(HookPart::Arity, message));
        };
        binding
            .body
            .set(Box::new(body))
            .map_err(|_| refused(HookPart::Bound, format!("the hook `{hook}` is bound already")))
    }
}

#[cfg(doctest)]
#[doc = include_str!("../../docs/hooks.md")]
struct HooksPage;
