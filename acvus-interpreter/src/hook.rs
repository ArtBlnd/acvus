//! A hook one program declares, bound by the host to another program's lent
//! entry, which runs within the call on the call's own arguments
//! (RFC-0101).

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use acvus_ast::Span;
use acvus_extern::{
    Args, ByArgs, ByOutput, Contribution, Ctx, Declared, ExternFn, ExternHandler, Finished, FnDecl,
    HandlerFactory, Holding, Instances, Laws, Manifest, Nth, Output, Owned, PolyVars, Reaches, Registry,
    RetFinished, Returns, TyArg, TyVarBound, glue, kind,
};
use acvus_mir::analysis::inst_info;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{Callee, Inst, InstKind, MirBody, MirModule, ValueId};
use acvus_mir::ty::{
    Effect, EffectTerm, Flows, Mutability, ParamTerm, Poly, PolyTy, Reissue, Ty, TypeArg,
};
use acvus_utils::{Astr, Interner};
use futures::FutureExt;
use rustc_hash::FxHashMap;

use crate::host::{
    Access, Cause, Compiled, HostError, Named, Origin, Program, Refusal, Storage, SyncAccess, span_of,
};
use crate::interpreter::Compilation;
use crate::machine::call_module;
use crate::port::{Gate, Port, end_run};
use crate::runtime::AcvusRuntime;
use crate::value::Value;

/// One table of names for several programs. A variant's tag and an object's
/// field names are interned words inside the values a program makes, so a
/// hook lends one program's values to another only where both were compiled
/// over the same table (`Host::with_names`).
#[derive(Clone, Default)]
pub struct Names(pub(crate) Interner);

impl Names {
    pub fn new() -> Self {
        Names(Interner::new())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookEffect {
    Pure,
    Idempotent,
    Opaque,
}

impl HookEffect {
    fn effect(self) -> Effect {
        match self {
            HookEffect::Pure => Effect::PURE,
            HookEffect::Idempotent => Effect::IDEMPOTENT,
            HookEffect::Opaque => Effect::OPAQUE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookPart {
    Argument(usize),
    Result,
    Arity,
    Effect,
    SeparateNames,
    Bound,
}

// -- Lent inputs ---------------------------------------------------------

/// The inputs of an entry a hook runs, in the order the hook's arguments
/// fill them. The entry reads each and never takes it.
#[derive(Default)]
pub struct LentInputs {
    fields: Vec<LentField>,
}

struct LentField {
    name: String,
    declared: fn(&Interner) -> PolyTy,
    reached_by: Option<Mutability>,
}

impl LentInputs {
    pub fn new() -> Self {
        LentInputs::default()
    }

    pub fn field<T>(self, name: &str) -> Self
    where
        T: Declared,
    {
        self.with(name, T::declared, None)
    }

    pub fn field_ref<T>(self, name: &str) -> Self
    where
        T: Declared,
    {
        self.with(name, T::declared, Some(Mutability::Shared))
    }

    pub fn field_mut<T>(self, name: &str) -> Self
    where
        T: Declared,
    {
        self.with(name, T::declared, Some(Mutability::Mut))
    }

    fn with(mut self, name: &str, declared: fn(&Interner) -> PolyTy, reached_by: Option<Mutability>) -> Self {
        self.fields.push(LentField {
            name: name.to_owned(),
            declared,
            reached_by,
        });
        self
    }

    pub(crate) fn resolved(&self, interner: &Interner) -> Vec<DeclaredInput> {
        self.fields
            .iter()
            .map(|field| {
                let target = (field.declared)(interner);
                let ty = match field.reached_by {
                    None => target,
                    Some(mutability) => PolyTy::Ref(mutability, Box::new(TypeArg::uniform(target))),
                };
                DeclaredInput {
                    name: interner.intern(&field.name),
                    ty,
                }
            })
            .collect()
    }
}

pub(crate) struct DeclaredInput {
    pub(crate) name: Astr,
    pub(crate) ty: PolyTy,
}

// -- Declaring a hook ----------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arity {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
}

impl Arity {
    const ALL: [Arity; 9] = [
        Arity::Zero,
        Arity::One,
        Arity::Two,
        Arity::Three,
        Arity::Four,
        Arity::Five,
        Arity::Six,
        Arity::Seven,
        Arity::Eight,
    ];

    fn of(count: usize) -> Option<Arity> {
        Arity::ALL.get(count).copied()
    }

    fn count(self) -> usize {
        self as usize
    }
}

pub(crate) struct HookDecl {
    pub(crate) name: String,
    arity: Arity,
    effect: HookEffect,
    slot: Arc<HookSlot>,
}

pub(crate) struct HookSlot {
    bound: OnceLock<Arc<Binding>>,
    name: String,
}

impl HookDecl {
    pub(crate) fn new(name: &str, arity: usize, effect: HookEffect) -> Option<HookDecl> {
        let arity = Arity::of(arity)?;
        Some(HookDecl {
            name: name.to_owned(),
            arity,
            effect,
            slot: Arc::new(HookSlot {
                bound: OnceLock::new(),
                name: name.to_owned(),
            }),
        })
    }

    pub(crate) fn qref(&self, interner: &Interner) -> QualifiedRef {
        QualifiedRef::root(interner.intern(&self.name))
    }

    pub(crate) fn registry(&self) -> Registry<AcvusRuntime> {
        let name = self.name.clone();
        let arity = self.arity;
        let effect = self.effect.effect();
        let slot = Arc::clone(&self.slot);
        Registry::new(move |interner: &Interner| {
            let mut contribution = Contribution::of(Manifest {
                types: Vec::new(),
                signatures: Vec::new(),
                fns: Vec::new(),
            });
            let declaring = Declaring {
                interner,
                name: &name,
                effect,
            };
            contribution.declare(declaring.at(arity, slot));
            contribution
        })
    }

    pub(crate) fn compiled(self, sites: Vec<HookSite>) -> CompiledHook {
        CompiledHook {
            slot: self.slot,
            arity: self.arity,
            effect: self.effect,
            sites,
        }
    }
}

struct Declaring<'a> {
    interner: &'a Interner,
    name: &'a str,
    effect: Effect,
}

macro_rules! owned {
    ($at:literal) => {
        Owned<AcvusRuntime>
    };
}

macro_rules! declared_at {
    ($declaring:ident, $slot:ident; $result:literal;) => {{
        let vars = PolyVars::fresh($result + 1, 0, 0, 0);
        let result = <Option<Nth<kind::Type, $result>> as TyArg>::poly_ty($declaring.interner, &vars);
        let handler = glue::<AcvusRuntime, _, (ByOutput,), RetFinished>(
            move |ctx: &mut Ctx<'_, AcvusRuntime>, (out,), ret| {
                let out: Output<'_, Owned<AcvusRuntime>, AcvusRuntime> = out.take();
                ret.put($slot.called(ctx, &[], out))
            },
        );
        $declaring.extern_fn(&vars, Vec::new(), result, handler)
    }};
    ($declaring:ident, $slot:ident; $result:literal; $($at:literal),+) => {{
        let vars = PolyVars::fresh($result + 1, 0, 0, 0);
        let params = vec![$(
            ParamTerm::<Poly>::new(
                $declaring.interner.intern(&format!("{}.{}", $declaring.name, $at)),
                <Nth<kind::Type, $at> as TyArg>::poly_ty($declaring.interner, &vars),
            )
        ),+];
        let result = <Option<Nth<kind::Type, $result>> as TyArg>::poly_ty($declaring.interner, &vars);
        let handler = glue::<AcvusRuntime, _, (ByArgs<($(owned!($at),)+)>, ByOutput), RetFinished>(
            move |ctx: &mut Ctx<'_, AcvusRuntime>, (args, out), ret| {
                let args: Args<'_, ($(owned!($at),)+), AcvusRuntime> = args.take();
                let out: Output<'_, Owned<AcvusRuntime>, AcvusRuntime> = out.take();
                // SAFETY: this is the runtime, and the capability reaches no
                // handler body: `called` copies the words into the entry's
                // frame, which takes none of them.
                let holding = unsafe { Holding::new() };
                ret.put($slot.called(ctx, args.lent_words(holding), out))
            },
        );
        $declaring.extern_fn(&vars, params, result, handler)
    }};
}

impl Declaring<'_> {
    fn at(&self, arity: Arity, slot: Arc<HookSlot>) -> ExternFn<AcvusRuntime> {
        match arity {
            Arity::Zero => declared_at!(self, slot; 0;),
            Arity::One => declared_at!(self, slot; 1; 0),
            Arity::Two => declared_at!(self, slot; 2; 0, 1),
            Arity::Three => declared_at!(self, slot; 3; 0, 1, 2),
            Arity::Four => declared_at!(self, slot; 4; 0, 1, 2, 3),
            Arity::Five => declared_at!(self, slot; 5; 0, 1, 2, 3, 4),
            Arity::Six => declared_at!(self, slot; 6; 0, 1, 2, 3, 4, 5),
            Arity::Seven => declared_at!(self, slot; 7; 0, 1, 2, 3, 4, 5, 6),
            Arity::Eight => declared_at!(self, slot; 8; 0, 1, 2, 3, 4, 5, 6, 7),
        }
    }

    fn extern_fn<H>(
        &self,
        vars: &PolyVars,
        params: Vec<ParamTerm<Poly>>,
        result: PolyTy,
        handler: H,
    ) -> ExternFn<AcvusRuntime>
    where
        H: HandlerFactory<AcvusRuntime> + 'static,
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
                copies: None,
                cost: None,
            },
            instances: Instances::generic(ExternHandler::sync(handler)),
        }
    }
}

// -- Call sites ----------------------------------------------------------

pub(crate) struct HookSite {
    origin: Option<Origin>,
    span: Option<Span>,
    args: Vec<Ty>,
    payload: Ty,
}

struct Called<'m> {
    hook: &'m QualifiedRef,
    args: Vec<Ty>,
    result: Ty,
}

fn settled(body: &MirBody, value: ValueId) -> &Ty {
    body.val_types
        .get(&value)
        .unwrap_or_else(|| panic!("the checker settles a type for every value, and {value:?} has none"))
}

fn extern_named<'m>(body: &'m MirBody, inst: &'m Inst) -> Option<Called<'m>> {
    match &inst.kind {
        InstKind::FunctionCall {
            dst,
            callee: Callee::Extern { id, .. },
            args,
            ..
        } => Some(Called {
            hook: id,
            args: args.iter().map(|arg| settled(body, *arg).clone()).collect(),
            result: settled(body, *dst).clone(),
        }),
        InstKind::Spawn {
            dst,
            callee: Callee::Extern { id, .. },
            args,
            ..
        } => {
            let Ty::Handle(spawned) = settled(body, *dst) else {
                panic!("a spawn's result is typed a handle, and {dst:?} is not");
            };
            Some(Called {
                hook: id,
                args: args.iter().map(|arg| settled(body, *arg).clone()).collect(),
                result: (**spawned).clone(),
            })
        }
        InstKind::LoadFunction { dst, id } => {
            let Ty::Fn { params, ret, .. } = settled(body, *dst) else {
                panic!("the value a `LoadFunction` makes is typed a function, and {dst:?} is not");
            };
            Some(Called {
                hook: id,
                args: params.iter().map(|param| param.ty.clone()).collect(),
                result: (**ret).clone(),
            })
        }
        _ => None,
    }
}

pub(crate) fn sites_of<O>(
    modules: &FxHashMap<QualifiedRef, MirModule>,
    hook: QualifiedRef,
    origin_of: O,
) -> Vec<HookSite>
where
    O: Fn(&QualifiedRef) -> Option<Origin>,
{
    let mut sites = Vec::new();
    for (qref, module) in modules {
        for body in std::iter::once(&module.main).chain(module.closures.values()) {
            for inst in &body.insts {
                let Some(Called { hook: named, args, result }) = extern_named(body, inst) else {
                    continue;
                };
                if *named != hook {
                    continue;
                }
                let Ty::Option(payload) = result else {
                    panic!(
                        "a call of the hook {hook:?} is typed {result:?}; a hook is declared \
                         `Option<T>`, which every call's result is"
                    );
                };
                sites.push(HookSite {
                    origin: origin_of(qref),
                    span: span_of(inst.span),
                    args,
                    payload: *payload,
                });
            }
        }
    }
    sites
}

pub(crate) struct CompiledHook {
    slot: Arc<HookSlot>,
    arity: Arity,
    effect: HookEffect,
    sites: Vec<HookSite>,
}

impl CompiledHook {
    pub(crate) fn is_bound(&self) -> bool {
        self.slot.bound.get().is_some()
    }
}

// -- Lent entries --------------------------------------------------------

pub(crate) struct LentEntry {
    by_argument: Vec<LentInput>,
    argument_of_param: Vec<usize>,
    effect: Effect,
}

struct LentInput {
    name: Astr,
    ty: Ty,
}

impl LentEntry {
    pub(crate) fn of(interner: &Interner, order: &[Astr], main: &MirBody, effect: Effect) -> LentEntry {
        let argument_of_param: Vec<usize> = main
            .params
            .iter()
            .map(|(param, _)| {
                order.iter().position(|name| name == param).unwrap_or_else(|| {
                    panic!(
                        "the lent entry's module takes `${}`, which its declared inputs lack",
                        interner.resolve(*param)
                    )
                })
            })
            .collect();
        assert_eq!(
            argument_of_param.len(),
            order.len(),
            "a lent entry's module takes each of its declared inputs once"
        );
        let inputs = order
            .iter()
            .map(|name| {
                let (_, value) = main
                    .params
                    .iter()
                    .find(|(param, _)| param == name)
                    .expect("every declared input is a parameter, as `argument_of_param` found");
                LentInput {
                    name: *name,
                    ty: settled(main, *value).clone(),
                }
            })
            .collect();
        LentEntry {
            by_argument: inputs,
            argument_of_param,
            effect,
        }
    }

    pub(crate) fn display(&self, interner: &Interner) -> String {
        let written: Vec<String> = self
            .by_argument
            .iter()
            .map(|input| format!("${}: {}", interner.resolve(input.name), input.ty.display(interner)))
            .collect();
        format!("({})", written.join(", "))
    }
}

enum Taken {
    Moved,
    Written,
}

struct OwningParam {
    name: Astr,
    value: ValueId,
}

fn taken(body: &MirBody, kind: &InstKind, param: ValueId) -> Option<Taken> {
    match kind {
        InstKind::Drop { src } if *src == param => None,
        InstKind::Ref {
            target, mutability, ..
        } if inst_info::storage(target) == Some(param) => match mutability {
            Mutability::Shared => None,
            Mutability::Mut => Some(Taken::Written),
        },
        InstKind::Take { dst, target, .. } if inst_info::storage(target) == Some(param) => {
            (settled(body, *dst).is_word() != Some(true)).then_some(Taken::Moved)
        }
        InstKind::Assign { target, .. } if inst_info::storage(target) == Some(param) => Some(Taken::Written),
        kind => inst_info::uses(kind).contains(&param).then_some(Taken::Moved),
    }
}

pub(crate) fn lent_module(
    interner: &Interner,
    entry: &str,
    origin: &Option<Origin>,
    module: &MirModule,
) -> Result<MirModule, Vec<Refusal>> {
    let main = &module.main;
    let owning: Vec<OwningParam> = main
        .params
        .iter()
        .filter(|(_, value)| settled(main, *value).is_word() != Some(true))
        .map(|(name, value)| OwningParam {
            name: *name,
            value: *value,
        })
        .collect();
    let refusals: Vec<Refusal> = main
        .insts
        .iter()
        .flat_map(|inst| {
            owning.iter().filter_map(move |param| {
                let taken = taken(main, &inst.kind, param.value)?;
                let name = interner.resolve(param.name);
                let what = match taken {
                    Taken::Moved => "moves",
                    Taken::Written => "writes",
                };
                let message = format!(
                    "the lent entry `{entry}` {what} its input `${name}`, which the hook's caller \
                     lends it: a lent input is read in place, through `&${name}` or as a copied \
                     word, and never taken (RFC-0101 rule 3)"
                );
                Some(Refusal {
                    span: span_of(inst.span),
                    ..Refusal::of(origin.clone(), message)
                })
            })
        })
        .collect();
    if !refusals.is_empty() {
        return Err(refusals);
    }
    let mut lent = module.clone();
    lent.main.insts.retain(|inst| {
        !matches!(&inst.kind, InstKind::Drop { src } if owning.iter().any(|param| param.value == *src))
    });
    Ok(lent)
}

pub(crate) fn lent_called<O>(
    modules: &FxHashMap<QualifiedRef, MirModule>,
    lent: &FxHashMap<QualifiedRef, String>,
    origin_of: O,
) -> Vec<Refusal>
where
    O: Fn(&QualifiedRef) -> Option<Origin>,
{
    let mut refusals = Vec::new();
    for (qref, module) in modules {
        for body in std::iter::once(&module.main).chain(module.closures.values()) {
            for inst in &body.insts {
                let (InstKind::FunctionCall {
                    callee: Callee::Direct(id),
                    ..
                }
                | InstKind::Spawn {
                    callee: Callee::Direct(id),
                    ..
                }
                | InstKind::LoadFunction { id, .. }) = &inst.kind
                else {
                    continue;
                };
                let Some(name) = lent.get(id) else {
                    continue;
                };
                let message = format!(
                    "the lent entry `{name}` is called here; a lent entry runs only from a hook, \
                     whose caller keeps what it lends (RFC-0101 rule 3)"
                );
                refusals.push(Refusal {
                    span: span_of(inst.span),
                    ..Refusal::of(origin_of(qref), message)
                });
            }
        }
    }
    refusals
}

// -- Binding -------------------------------------------------------------

type Run = dyn for<'c> Fn(Call<'c>) -> Ran<'c> + Send + Sync;

struct Binding {
    callee: Arc<Compiled>,
    entry: QualifiedRef,
    entry_name: String,
    argument_of_param: Vec<usize>,
    run: Box<Run>,
}

impl<A> Program<A>
where
    A: Access,
{
    pub fn bind<F>(&self, hook: &str, lent: &Lent, run: F) -> Result<(), HostError>
    where
        F: for<'c> Fn(Call<'c>) -> Ran<'c> + Send + Sync + 'static,
    {
        let caller = &self.compiled;
        let Some(declared) = caller.hooks.get(hook) else {
            return Err(HostError::NotInGraph {
                what: Named::Hook(hook.to_owned()),
            });
        };
        let callee = &lent.callee;
        let entry = lent.entry.as_str();
        let lent = callee.lent_entry(entry)?;
        let refused = |part: HookPart, message: String| Refusal {
            cause: Some(Cause::Hook {
                hook: hook.to_owned(),
                part,
            }),
            ..Refusal::of(None, message)
        };
        if declared.is_bound() {
            let message = format!("the hook `{hook}` is bound already");
            return Err(HostError::Refused(vec![refused(HookPart::Bound, message)]));
        }
        let site_names = caller.interner();
        let entry_names = callee.interner();
        if site_names.id() != entry_names.id() {
            let message = format!(
                "the hook `{hook}` and the entry `{entry}` were compiled over two tables of names, \
                 and a value holds its program's names; compile both with `Host::with_names` over \
                 one `Names` (RFC-0101 rule 3)"
            );
            return Err(HostError::Refused(vec![refused(HookPart::SeparateNames, message)]));
        }
        let lent_inputs = &lent.shape.by_argument;
        if declared.arity.count() != lent_inputs.len() {
            let message = format!(
                "the hook `{hook}` takes {} arguments, and the entry `{entry}` takes {} inputs {}",
                declared.arity.count(),
                lent_inputs.len(),
                lent.shape.display(entry_names)
            );
            return Err(HostError::Refused(vec![refused(HookPart::Arity, message)]));
        }

        let mut refusals = Vec::new();
        let seen = as_the_caller_sees(&lent.shape.effect);
        let allowed = declared.effect.effect();
        if !seen.at_most(&allowed) || lent.suspends {
            let waits = match lent.suspends {
                true => " and can wait",
                false => "",
            };
            let message = format!(
                "the entry `{entry}` runs at {seen}{waits}, past the hook `{hook}`'s declared \
                 {allowed}: the hook's caller sees a read of the entry's context as an idempotent \
                 call and a write as an opaque one, and the entry runs within a call that cannot \
                 wait (RFC-0101 rule 4)"
            );
            refusals.push(refused(HookPart::Effect, message));
        }
        for site in &declared.sites {
            let at_site = |part: HookPart, message: String| Refusal {
                span: site.span,
                origin: site.origin.clone(),
                ..refused(part, message)
            };
            let called = site_name(&site.origin);
            assert_eq!(
                site.args.len(),
                lent_inputs.len(),
                "a call of the hook `{hook}` passes the arguments its declaration takes"
            );
            for (at, (arg, input)) in site.args.iter().zip(lent_inputs).enumerate() {
                let site_arg = Typed {
                    names: site_names,
                    ty: arg,
                };
                let entry_input = Typed {
                    names: entry_names,
                    ty: &input.ty,
                };
                if let Err(unfit) = site_arg.fits(entry_input, Level::Argument) {
                    let message = format!(
                        "argument {at} of the call of the hook `{hook}` in {called} is {}, and the \
                         entry `{entry}` takes `${}: {}`{}",
                        arg.display(site_names),
                        entry_names.resolve(input.name),
                        input.ty.display(entry_names),
                        unfit.why()
                    );
                    refusals.push(at_site(HookPart::Argument(at), message));
                }
            }
            let site_result = Typed {
                names: site_names,
                ty: &site.payload,
            };
            let entry_result = Typed {
                names: entry_names,
                ty: lent.ret,
            };
            if let Err(unfit) = site_result.fits(entry_result, Level::Result) {
                let message = format!(
                    "the call of the hook `{hook}` in {called} settles its result as Option<{}>, \
                     and the entry `{entry}` returns {}{}",
                    site.payload.display(site_names),
                    lent.ret.display(entry_names),
                    unfit.why()
                );
                refusals.push(at_site(HookPart::Result, message));
            }
        }
        if !refusals.is_empty() {
            return Err(HostError::Refused(refusals));
        }

        let binding = Binding {
            callee: Arc::clone(callee),
            entry: lent.qref,
            entry_name: entry.to_owned(),
            argument_of_param: lent.shape.argument_of_param.clone(),
            run: Box::new(run),
        };
        if declared.slot.bound.set(Arc::new(binding)).is_err() {
            let message = format!("the hook `{hook}` is bound already");
            return Err(HostError::Refused(vec![refused(HookPart::Bound, message)]));
        }
        Ok(())
    }
}

/// A lent entry of a program compiled for synchronous access, which a hook
/// of another program may be bound to.
pub struct Lent {
    callee: Arc<Compiled>,
    entry: String,
}

impl Program<SyncAccess> {
    pub fn lent(&self, entry: &str) -> Result<Lent, HostError> {
        self.compiled.lent_entry(entry)?;
        Ok(Lent {
            callee: Arc::clone(&self.compiled),
            entry: entry.to_owned(),
        })
    }
}

pub(crate) struct FoundLent<'p> {
    pub(crate) qref: QualifiedRef,
    pub(crate) shape: &'p LentEntry,
    pub(crate) ret: &'p Ty,
    pub(crate) suspends: bool,
}

fn site_name(origin: &Option<Origin>) -> String {
    match origin {
        Some(Origin::Entry(name)) => format!("`{name}`"),
        Some(Origin::Init(key)) => format!("the init of `@{key}`"),
        Some(Origin::Binding(name)) => format!("the binding of `${name}`"),
        None => "the program".to_owned(),
    }
}

/// The entry's effect as the hook's caller sees it: the caller's checker
/// cannot see the entry's contexts, so a read of one is ordered as an
/// idempotent call is, and a write as an opaque one.
fn as_the_caller_sees(effect: &Effect) -> Effect {
    let touched = [
        (!effect.reads.is_empty()).then_some(Reissue::Idempotent),
        (!effect.writes.is_empty()).then_some(Reissue::Opaque),
    ];
    let reissue = touched.into_iter().flatten().fold(effect.reissue, Reissue::max);
    let untouched = effect.reads.is_empty() && effect.writes.is_empty();
    Effect::new(reissue, effect.commutes && untouched).at_task(effect.task)
}

// -- Structure -----------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Argument,
    Inner,
    Result,
}

enum Unfit {
    Differs,
    Extension,
    Function,
    Task,
    View,
    InnerReference,
    Laid,
}

impl Unfit {
    fn why(&self) -> &'static str {
        match self {
            Unfit::Differs => "",
            Unfit::Extension => {
                "; an extension type is its registry's, and only the language's own types cross a hook"
            }
            Unfit::Function => "; a function value runs its program's code, and crosses no hook",
            Unfit::Task => "; a task handle runs the code that made it, and crosses no hook",
            Unfit::View => "; a view is two of its lender's registers, and crosses no hook",
            Unfit::InnerReference => {
                "; a reference crosses a hook only as a whole argument, where the entry cannot \
                 hand the caller a loan its checker never saw"
            }
            Unfit::Laid => "; a reference to a specialized layout crosses no hook",
        }
    }
}

#[derive(Clone, Copy)]
struct Typed<'a> {
    names: &'a Interner,
    ty: &'a Ty,
}

impl<'a> Typed<'a> {
    fn at(self, ty: &'a Ty) -> Typed<'a> {
        Typed { names: self.names, ty }
    }

    fn fits(self, other: Typed<'_>, level: Level) -> Result<(), Unfit> {
        carried(self.ty, level)?;
        carried(other.ty, level)?;
        match self.same(other) {
            true => Ok(()),
            false => Err(Unfit::Differs),
        }
    }

    /// An object's field set records how its type was learned and is not
    /// compared: two objects of the same fields are laid out alike whatever
    /// their sets.
    fn same(self, other: Typed<'_>) -> bool {
        let names = |held: Astr| self.names.resolve(held);
        let other_names = |held: Astr| other.names.resolve(held);
        match (self.ty, other.ty) {
            (Ty::Int(x), Ty::Int(y)) => x == y,
            (Ty::Float, Ty::Float)
            | (Ty::Char, Ty::Char)
            | (Ty::String, Ty::String)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Unit, Ty::Unit)
            | (Ty::Never, Ty::Never) => true,
            (Ty::Array(x, n), Ty::Array(y, m)) => n == m && self.at(x).same(other.at(y)),
            (Ty::Option(x), Ty::Option(y)) => self.at(x).same(other.at(y)),
            (Ty::Result(xo, xe), Ty::Result(yo, ye)) => {
                self.at(xo).same(other.at(yo)) && self.at(xe).same(other.at(ye))
            }
            (Ty::Tuple(xs), Ty::Tuple(ys)) => {
                xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| self.at(x).same(other.at(y)))
            }
            (Ty::Object(xs), Ty::Object(ys)) => {
                xs.len() == ys.len()
                    && xs.iter().all(|(name, x)| {
                        ys.iter()
                            .find(|(held, _)| other_names(**held) == names(*name))
                            .is_some_and(|(_, y)| self.at(x).same(other.at(y)))
                    })
            }
            (
                Ty::Enum {
                    name: x_name,
                    variants: xs,
                    ..
                },
                Ty::Enum {
                    name: y_name,
                    variants: ys,
                    ..
                },
            ) => {
                let spelled = |x: Option<Astr>, y: Option<Astr>| x.map(names) == y.map(other_names);
                spelled(x_name.host, y_name.host)
                    && spelled(x_name.namespace, y_name.namespace)
                    && names(x_name.name) == other_names(y_name.name)
                    && xs.len() == ys.len()
                    && xs.iter().all(|(tag, x)| {
                        ys.iter()
                            .find(|(held, _)| other_names(**held) == names(*tag))
                            .is_some_and(|(_, y)| match (x, y) {
                                (None, None) => true,
                                (Some(x), Some(y)) => self.at(x).same(other.at(y)),
                                (None, Some(_)) | (Some(_), None) => false,
                            })
                    })
            }
            (Ty::Ref(xm, x), Ty::Ref(ym, y)) => match (&**x, &**y) {
                (TypeArg::Uniform(x), TypeArg::Uniform(y)) => xm == ym && self.at(x).same(other.at(y)),
                _ => false,
            },
            _ => false,
        }
    }
}

fn carried(ty: &Ty, level: Level) -> Result<(), Unfit> {
    let inner = |ty: &Ty| carried(ty, Level::Inner);
    match ty {
        Ty::Int(_) | Ty::Float | Ty::Char | Ty::String | Ty::Bool | Ty::Unit | Ty::Never => Ok(()),
        Ty::Array(element, _) | Ty::Option(element) => inner(element),
        Ty::Result(ok, err) => inner(ok).and_then(|()| inner(err)),
        Ty::Tuple(items) => items.iter().try_for_each(inner),
        Ty::Object(object) => object.values().try_for_each(inner),
        Ty::Enum { variants, .. } => variants.values().flatten().try_for_each(|payload| inner(payload)),
        Ty::Ref(_, target) => match (level, &**target) {
            (Level::Argument, TypeArg::Uniform(target)) => inner(target),
            (Level::Argument, _) => Err(Unfit::Laid),
            (Level::Inner | Level::Result, _) => Err(Unfit::InnerReference),
        },
        Ty::UserDefined { .. } => Err(Unfit::Extension),
        Ty::Fn { .. } => Err(Unfit::Function),
        Ty::Handle(_) => Err(Unfit::Task),
        Ty::Slice(_) | Ty::Str => Err(Unfit::View),
        Ty::Order | Ty::Error(_) | Ty::Var(_) => Err(Unfit::Differs),
    }
}

// -- Running -------------------------------------------------------------

thread_local! {
    /// A hook's handler is declared synchronous, `bind` admits only an entry
    /// that cannot wait, and `Call` stays on its thread, so the hook calls
    /// on this thread's stack are the call stack of the run that made them.
    static CALLERS: RefCell<Vec<Compilation>> = const { RefCell::new(Vec::new()) };
}

struct Calling;

impl Calling {
    fn enter(caller: Compilation, binding: &Binding, hook: &str) -> Calling {
        let callee = binding.callee.shared.compilation;
        let reentered = CALLERS.with_borrow_mut(|callers| {
            let reentered = callee == caller || callers.contains(&callee);
            if !reentered {
                callers.push(caller);
            }
            reentered
        });
        if reentered {
            panic!(
                "the hook `{hook}` would run the entry `{}` of a program already running on this \
                 call stack (RFC-0101 rule 4)",
                binding.entry_name
            );
        }
        Calling
    }
}

impl Drop for Calling {
    fn drop(&mut self) {
        CALLERS.with_borrow_mut(|callers| {
            callers.pop();
        });
    }
}

impl HookSlot {
    /// An unbound hook ends the run with `HostError::Unbound`: `Page`'s runs
    /// refuse such a program before they start, and a load's init, which
    /// runs outside a run, meets the hook here.
    fn called<'c>(
        &self,
        ctx: &mut Ctx<'_, AcvusRuntime>,
        words: &[Owned<AcvusRuntime>],
        out: Output<'c, Owned<AcvusRuntime>, AcvusRuntime>,
    ) -> Finished<'c, Owned<AcvusRuntime>, AcvusRuntime> {
        let Some(binding) = self.bound.get() else {
            end_run(HostError::Unbound {
                hook: self.name.clone(),
            })
        };
        let _calling = Calling::enter(ctx.rt.shared.compilation, binding, &self.name);
        assert_eq!(
            words.len(),
            binding.argument_of_param.len(),
            "a call of the hook `{}` passes the arguments its bound entry takes, as `bind` found",
            self.name
        );
        let arguments = binding.argument_of_param.iter().map(|at| *words[*at]).collect();
        let call = Call {
            binding: Arc::clone(binding),
            arguments,
            out,
            not_send: PhantomData,
        };
        (binding.run)(call).finished
    }
}

pub struct Call<'c> {
    binding: Arc<Binding>,
    arguments: Vec<Value>,
    out: Output<'c, Owned<AcvusRuntime>, AcvusRuntime>,
    not_send: PhantomData<*const ()>,
}

pub struct Ran<'c> {
    finished: Finished<'c, Owned<AcvusRuntime>, AcvusRuntime>,
}

impl<'c> Call<'c> {
    /// # Panics
    /// The entry traps, its storage refuses, or its program has an unbound
    /// hook: the calling run ends there, with that error.
    pub fn run<S>(self, storage: &mut S) -> Ran<'c>
    where
        S: Storage,
    {
        let Call {
            binding,
            arguments,
            out,
            ..
        } = self;
        let callee = &binding.callee;
        if let Some(hook) = callee.unbound_hook() {
            end_run(HostError::Unbound { hook: hook.to_owned() });
        }
        let storage: &mut dyn Storage = storage;
        // SAFETY: `closing` closes the gate when this function returns or
        // unwinds, while `storage` is still borrowed exclusively.
        let port = Port::gate(unsafe { Gate::open(storage) });
        let closing = Closing(Arc::clone(&port));
        let rt = callee.shared.runtime_over(Arc::clone(&port));
        let ran = call_module::<Value>(rt, binding.entry, arguments).now_or_never();
        drop(closing);
        let Some(value) = ran else {
            panic!(
                "the entry `{}` waited, and `bind` admits only an entry whose prepared body cannot",
                binding.entry_name
            );
        };
        // SAFETY: the run moved its result out to this caller, and no other
        // holder owns it.
        let value = unsafe { Owned::from_value(Holding::new(), value) };
        // SAFETY: `bind` found the entry's result the type every call site
        // of the hook settled for its `Option`'s payload.
        let finished = unsafe { out.finish_whole(value) };
        Ran { finished }
    }
}

struct Closing(Arc<Port>);

impl Drop for Closing {
    fn drop(&mut self) {
        self.0.close();
    }
}

#[cfg(doctest)]
#[doc = include_str!("../../docs/hooks.md")]
struct HooksPage;
