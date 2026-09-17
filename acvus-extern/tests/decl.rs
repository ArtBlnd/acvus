//! RFC-0009 at its contract: one Rust signature yields the acvus type and
//! a handler for any runtime. `Tiny` is a runtime written for this test
//! alone, so nothing here depends on an interpreter.

use std::any::Any;
use std::future::Ready;
use std::marker::PhantomData;
use std::sync::Arc;

use acvus_extern::{
    Arr, CallToken, ClosureFn, Cross, Eff, Effect, EffectTerm, EffectVar, ExternFn, ExternHandler,
    ExternType, Externs, Fn1, HasInstance, Interner, LenTerm, LenVar, PolyTy, Pure, Registry,
    Runtime, TyArg, TyVar, TypeArg, TypeRegistry, TypesOnly, extern_fn, extern_registry,
    extern_signature,
};

// -- A runtime for this test ------------------------------------------

/// A value is a Rust value boxed whole, a closure, or a reference to
/// another value. `Erased` never compares equal: the test runtime has no
/// view into what it holds.
#[derive(Debug, Default)]
enum V {
    /// The value a handler took out of its argument slot.
    #[default]
    Taken,
    Erased(Box<dyn Any + Send + Sync>),
    Closure(Closure),
    Reference(*const V),
}

// SAFETY: a `Reference` is used only while its target is live (RFC-0018).
unsafe impl Send for V {}
unsafe impl Sync for V {}

impl PartialEq for V {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (V::Closure(a), V::Closure(b)) => Arc::ptr_eq(&a.0, &b.0),
            (V::Reference(a), V::Reference(b)) => std::ptr::eq(*a, *b),
            _ => false,
        }
    }
}

#[derive(Clone)]
struct Closure(Arc<dyn Fn(Vec<V>) -> V + Send + Sync>);

impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "closure")
    }
}

fn erased<T>(value: T) -> V
where
    T: Send + Sync + 'static,
{
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<V>() {
        let boxed: Box<dyn std::any::Any> = Box::new(value);
        return *boxed.downcast::<V>().expect("T is V");
    }
    V::Erased(Box::new(value))
}

/// A copy of the Rust value inside a lent `Erased`, or a panic naming the
/// mismatch.
fn peek<T>(value: &V) -> T
where
    T: Clone + Send + Sync + 'static,
{
    match value {
        V::Erased(any) => any
            .downcast_ref::<T>()
            .unwrap_or_else(|| panic!("peek: value is not a {}", std::any::type_name::<T>()))
            .clone(),
        V::Closure(_) => panic!(
            "peek: value is a closure, not a {}",
            std::any::type_name::<T>()
        ),
        V::Reference(_) => panic!(
            "peek: value is a reference, not a {}",
            std::any::type_name::<T>()
        ),
        V::Taken => panic!("peek: the value was already taken out of its slot"),
    }
}

/// The Rust value inside an `Erased`, or a panic naming the mismatch.
fn open<T>(value: V) -> T
where
    T: Send + Sync + 'static,
{
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<V>() {
        let boxed: Box<dyn std::any::Any> = Box::new(value);
        return *boxed.downcast::<T>().expect("T is V");
    }
    match value {
        V::Erased(any) => match any.downcast::<T>() {
            Ok(v) => *v,
            Err(_) => panic!("materialize: value is not a {}", std::any::type_name::<T>()),
        },
        V::Closure(_) => panic!(
            "materialize: value is a closure, not a {}",
            std::any::type_name::<T>()
        ),
        V::Reference(_) => panic!(
            "materialize: value is a reference, not a {}",
            std::any::type_name::<T>()
        ),
        V::Taken => panic!("materialize: the value was already taken out of its slot"),
    }
}

#[derive(Clone)]
struct Tiny;

impl Tiny {
    fn call(&self, f: &V, args: Vec<V>) -> V {
        match f {
            V::Closure(c) => (c.0)(args),
            V::Taken | V::Erased(_) | V::Reference(_) => {
                panic!("call on a value that is not a closure")
            }
        }
    }
}

/// The value a reference names.
fn target(reference: &V) -> &V {
    match reference {
        // SAFETY: the target is live for as long as the reference is used.
        V::Reference(p) => unsafe { &**p },
        other => panic!("not a reference: {other:?}"),
    }
}

/// The Rust value inside an `Erased`, or a panic naming the mismatch.
fn open_ref<T>(value: &V) -> &T
where
    T: Send + Sync + 'static,
{
    match value {
        V::Erased(any) => any
            .downcast_ref::<T>()
            .unwrap_or_else(|| panic!("open_ref: value is not a {}", std::any::type_name::<T>())),
        other => panic!("open_ref: not a value: {other:?}"),
    }
}

fn open_mut<T>(value: &mut V) -> &mut T
where
    T: Send + Sync + 'static,
{
    match value {
        V::Erased(any) => any
            .downcast_mut::<T>()
            .unwrap_or_else(|| panic!("open_mut: value is not a {}", std::any::type_name::<T>())),
        other => panic!("open_mut: not a value: {other:?}"),
    }
}

static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

impl acvus_extern::Cross<Tiny> for V {
    fn erase(self, _: &Tiny) -> V {
        self
    }

    unsafe fn materialize(_: &Tiny, value: V) -> Self {
        value
    }

    unsafe fn deref<'a>(_: &Tiny, reference: &'a V) -> &'a V {
        target(reference)
    }

    unsafe fn deref_mut<'a>(_: &Tiny, reference: &'a V) -> &'a mut V {
        let V::Reference(p) = reference else {
            panic!("deref_mut: not a reference: {reference:?}")
        };
        // SAFETY: the target is live and, by the checker, exclusively named.
        unsafe { &mut *(*p as *mut V) }
    }
}

impl acvus_extern::FromValue<Tiny> for V {
    fn from_value(_: &Tiny, value: V) -> V {
        value
    }
}

impl Runtime for Tiny {
    fn type_of(&self, value: &V) -> Option<std::any::TypeId> {
        let V::Erased(any) = value else {
            return None;
        };
        Some((**any).type_id())
    }
    fn type_name_of(&self, _: &V) -> Option<&'static str> {
        None
    }
    unsafe fn inline_ref<T>(value: &V) -> &T
    where
        T: acvus_extern::Inline,
    {
        open_ref::<T>(value)
    }
    unsafe fn inline_mut<T>(value: &mut V) -> &mut T
    where
        T: acvus_extern::Inline,
    {
        open_mut::<T>(value)
    }

    fn symbol(&self, name: &str) -> acvus_extern::Astr {
        SYMBOLS.intern(name)
    }

    type Value = V;
    type CallFuture<'a> = Ready<V>;

    unsafe fn materialize<T>(&self, value: V) -> T
    where
        T: Send + Sync + 'static,
    {
        open(value)
    }
    unsafe fn erase<T>(&self, value: T) -> V
    where
        T: Send + Sync + 'static,
    {
        erased(value)
    }
    unsafe fn value_as_ref<'a, T>(&'a self, value: &'a V) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        open_ref(value)
    }
    unsafe fn value_as_mut<'a, T>(&'a self, value: &'a mut V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        open_mut(value)
    }
    unsafe fn deref<'a, T>(&self, reference: &'a V) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        open_ref(target(reference))
    }
    unsafe fn deref_mut<'a, T>(&self, reference: &'a V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        let V::Reference(p) = reference else {
            panic!("deref_mut: not a reference: {reference:?}")
        };
        // SAFETY: the target is live and, by the checker, exclusively named.
        open_mut(unsafe { &mut *(*p as *mut V) })
    }
    unsafe fn reference(&self, target: &V) -> V {
        V::Reference(target as *const V)
    }
    fn call_is_sync(&self, _: &V) -> bool {
        true
    }
    fn call_now(&self, f: &V, args: &mut [V], _: CallToken) -> V {
        self.call(f, args.iter_mut().map(std::mem::take).collect())
    }
    fn call_0<'a>(&'a self, f: &'a V, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, Vec::new()))
    }
    fn call_1<'a>(&'a self, f: &'a V, a: V, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, vec![a]))
    }
    fn call_n<'a>(&'a self, f: &'a V, args: &mut [V], _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, args.iter_mut().map(std::mem::take).collect()))
    }
}

// -- Declarations under test ------------------------------------------

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Box")]
struct Boxed<T, E, Rt>(Vec<Rt::Value>, PhantomData<(T, E)>)
where
    T: TyVar,
    E: EffectVar,
    Rt: Runtime;

/// A value that is a source of its own: it carries an identity, so it moves.
#[derive(ExternType)]
#[repr(transparent)]
struct Token<I>(i64, PhantomData<I>)
where
    I: acvus_extern::IdentityVar;

#[derive(TyArg, Debug, PartialEq)]
struct Point {
    x: i64,
    label: String,
}

#[extern_fn(effect = pure)]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[extern_fn(name = "id_any", effect = E)]
fn identity<T, E>(v: T) -> T
where
    T: TyVar,
    E: EffectVar,
{
    v
}

#[extern_fn(effect = pure)]
async fn apply<T, U, E, Rt>(rt: &Rt, v: Boxed<T, E, Rt>, f: Fn1<T, U, E, Rt>) -> Boxed<U, E, Rt>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    let f = f.erased();
    let mut out = Vec::with_capacity(v.0.len());
    for item in v.0 {
        out.push(f.call(rt, (item,)).await);
    }
    Boxed(out, PhantomData)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn boxed<T, N, Rt>(rt: &Rt, items: Arr<T, N>) -> Boxed<T, Pure, Rt>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    Boxed(
        items
            .0
            .into_iter()
            .map(|v| unsafe { rt.erase::<T>(v) })
            .collect(),
        PhantomData,
    )
}

#[extern_fn]
async fn fetch(p: Point) -> Point {
    Point {
        x: p.x * 2,
        label: p.label,
    }
}

#[extern_fn(effect = idempotent)]
fn take_token<I>(t: Token<I>) -> i64
where
    I: acvus_extern::IdentityVar,
{
    t.0
}

/// A fresh draw: two draws in either order are the same program.
#[extern_fn(effect = idempotent, commutative)]
fn draw() -> i64 {
    4
}

/// Adds `by` to the lent place and returns the new value.
#[extern_fn(effect = pure)]
fn bump(n: &mut i64, by: i64) -> i64 {
    *n += by;
    *n
}

extern_signature! { ns: "t", fn eq<T>(a: &T, b: &T) -> bool where T: TyVar; }

#[extern_fn(instance_of = eq, effect = pure)]
fn eq_int(a: &i64, b: &i64) -> bool {
    a == b
}

#[extern_fn(instance_of = eq, effect = pure)]
fn eq_point(a: &Point, b: &Point) -> bool {
    a == b
}

/// Requires `eq` of its element type.
#[extern_fn(effect = pure)]
fn same<T, R>(rt: &R, a: T, b: T) -> Boxed<T, Pure, R>
where
    T: TyVar + HasInstance<eq>,
    R: Runtime,
{
    Boxed(
        vec![unsafe { rt.erase::<T>(a) }, unsafe { rt.erase::<T>(b) }],
        PhantomData,
    )
}

/// A greeting held by the handler: what a `#[state]` parameter carries.
struct Greeting(String);

#[extern_fn(effect = pure)]
fn greet(#[state] greeting: &Greeting, name: String) -> String {
    format!("{}, {name}", greeting.0)
}

fn registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "t",
        types: [Boxed<_, _, R>, Token<_>],
        signatures: [eq],
        fns: [add, identity, apply, boxed, fetch, take_token, draw, bump, eq_int, eq_point, same,
              greet(Greeting("hello".to_string()))],
    }
}

fn combined<R>() -> (Interner, Externs<R>)
where
    R: Runtime,
{
    let i = Interner::new();
    let reg = Externs::combine(vec![registry::<R>()], &i).expect("registries combine");
    (i, reg)
}

fn qref(i: &Interner, name: &str) -> acvus_extern::QualifiedRef {
    acvus_extern::QualifiedRef::qualified(i.intern("t"), i.intern(name))
}

/// The three parts of a function type this file asserts on.
struct FnShape {
    params: Vec<PolyTy>,
    ret: PolyTy,
    effect: EffectTerm<acvus_extern::Poly>,
}

fn fn_ty(ty: &PolyTy) -> FnShape {
    let PolyTy::Fn {
        params,
        ret,
        effect,
        ..
    } = ty
    else {
        panic!("not a function type: {ty:?}");
    };
    FnShape {
        params: params.iter().map(|p| p.ty.clone()).collect(),
        ret: (**ret).clone(),
        effect: effect.clone(),
    }
}

fn find<'a>(reg: &'a [acvus_extern::Function], i: &Interner, name: &str) -> &'a PolyTy {
    &reg.iter()
        .find(|f| i.resolve(f.qref.name) == name)
        .expect(name)
        .ty
}

#[test]
fn concrete_signature_and_declared_effect() {
    let (i, reg) = combined::<TypesOnly>();
    let add = fn_ty(find(&reg.functions, &i, "add"));
    assert_eq!(add.params, vec![PolyTy::I64, PolyTy::I64]);
    assert_eq!(add.ret, PolyTy::I64);
    assert_eq!(add.effect, EffectTerm::Known(Effect::PURE));

    assert_eq!(
        fn_ty(find(&reg.functions, &i, "fetch")).effect,
        EffectTerm::Known(Effect::OPAQUE)
    );
    assert_eq!(
        fn_ty(find(&reg.functions, &i, "take_token")).effect,
        EffectTerm::Known(Effect::IDEMPOTENT)
    );
    assert_eq!(
        fn_ty(find(&reg.functions, &i, "draw")).effect,
        EffectTerm::Known(Effect::IDEMPOTENT.commutative())
    );
}

#[test]
fn generic_parameters_become_positional_variables() {
    let (i, reg) = combined::<TypesOnly>();

    let id = fn_ty(find(&reg.functions, &i, "id_any"));
    assert_eq!(id.params, vec![PolyTy::Var(0)]);
    assert_eq!(id.ret, PolyTy::Var(0));
    assert_eq!(id.effect, EffectTerm::Var(0));

    let apply = fn_ty(find(&reg.functions, &i, "apply"));
    let boxed_of = |t: PolyTy| PolyTy::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Box")),
        type_args: vec![TypeArg::uniform(t)],
        effect_args: vec![EffectTerm::Var(0)],
        identity_args: vec![],
    };
    assert_eq!(apply.params[0], boxed_of(PolyTy::Var(0)));
    let PolyTy::Fn {
        params: fparams,
        ret: fret,
        effect,
        ..
    } = &apply.params[1]
    else {
        panic!("callback is a function type");
    };
    assert_eq!(fparams[0].ty, PolyTy::Var(0));
    assert_eq!(**fret, PolyTy::Var(1));
    assert_eq!(*effect, EffectTerm::Var(0));
    assert_eq!(apply.ret, boxed_of(PolyTy::Var(1)));

    let boxed = fn_ty(find(&reg.functions, &i, "boxed"));
    assert_eq!(
        boxed.params,
        vec![PolyTy::Array(Box::new(PolyTy::Var(0)), LenTerm::Var(0))]
    );
    assert_eq!(
        boxed.ret,
        PolyTy::UserDefined {
            id: acvus_extern::QualifiedRef::root(i.intern("Box")),
            type_args: vec![TypeArg::uniform(PolyTy::Var(0))],
            effect_args: vec![EffectTerm::Known(Effect::PURE)],
            identity_args: vec![],
        }
    );
}

#[test]
fn types_and_casts_reach_the_type_registry() {
    let (i, reg) = combined::<TypesOnly>();
    let tr = &reg.types;
    let decl = tr.get(acvus_extern::QualifiedRef::root(i.intern("Box")));
    assert_eq!(decl.type_params.len(), 1);
    assert_eq!(decl.effect_params, 1);
    let token = tr.get(acvus_extern::QualifiedRef::root(i.intern("Token")));
    assert_eq!(token.type_params.len(), 0);
    assert_eq!(token.identity_params, 1);

    let rules = tr.rules_to(acvus_extern::QualifiedRef::root(i.intern("Box")));
    assert_eq!(rules.len(), 1);
    assert_eq!(i.resolve(rules[0].fn_ref.name), "boxed");
    assert_eq!(
        rules[0].from,
        PolyTy::Array(Box::new(PolyTy::Var(0)), LenTerm::Var(0))
    );
}

fn call_sync(handler: &ExternHandler<Tiny>, mut args: Vec<V>) -> V {
    match handler {
        ExternHandler::Sync(f) => f(&Tiny, &mut args),
        ExternHandler::Async(_) => panic!("expected a sync handler"),
    }
}

#[test]
fn a_borrowed_parameter_is_a_reference_type_and_writes_through() {
    let (i, reg) = combined::<Tiny>();
    let PolyTy::Fn { params, .. } = find(&reg.functions, &i, "bump") else {
        panic!("bump is a function");
    };
    assert_eq!(
        params[0].ty,
        PolyTy::Ref(
            acvus_extern::Mutability::Mut,
            Box::new(TypeArg::uniform(PolyTy::I64))
        )
    );
    assert_eq!(params[1].ty, PolyTy::I64);

    let h = handler(&reg, &i, "bump");
    let place = erased(40i64);
    let r = call_sync(h, vec![unsafe { Tiny.reference(&place) }, erased(2i64)]);
    assert_eq!(open::<i64>(r), 42);
    assert_eq!(
        open::<i64>(place),
        42,
        "the place the reference named was written"
    );
}

async fn call_async(handler: &ExternHandler<Tiny>, mut args: Vec<V>) -> V {
    match handler {
        ExternHandler::Async(f) => f(Tiny, &mut args).await,
        ExternHandler::Sync(_) => panic!("expected an async handler"),
    }
}

fn handler<'a>(reg: &'a Externs<Tiny>, i: &Interner, name: &str) -> &'a ExternHandler<Tiny> {
    match reg.handlers[&qref(i, name)].as_slice() {
        [only] => only,
        many => panic!("{name} has {} instances; pick one by call type", many.len()),
    }
}

/// The handler the compiler would settle a call of type `callee_ty` on:
/// the concrete instance whose signature the type matches, else the
/// generic one. The instance list the compiler reads and the handler list
/// the runtime indexes are the same list.
fn instance_for<'a>(
    reg: &'a Externs<Tiny>,
    i: &Interner,
    name: &str,
    callee_ty: &acvus_extern::Ty,
) -> Option<&'a ExternHandler<Tiny>> {
    let function = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(i, name))
        .expect("declared");
    let acvus_extern::FnKind::Extern { instances, .. } = &function.kind else {
        panic!("{name} is extern")
    };
    let handlers = &reg.handlers[&qref(i, name)];
    let concrete = instances
        .concrete
        .iter()
        .position(|sig| acvus_mir::ty::matches_poly(callee_ty, sig));
    let index = concrete.or_else(|| instances.generic.then(|| instances.generic_index()))?;
    Some(&handlers[index])
}

#[tokio::test]
async fn handlers_run_the_rust_body_on_the_test_runtime() {
    let (i, reg) = combined::<Tiny>();

    assert_eq!(
        open::<i64>(call_sync(
            handler(&reg, &i, "add"),
            vec![erased(40i64), erased(2i64)]
        )),
        42
    );
    assert_eq!(
        open::<&str>(open::<V>(call_sync(
            handler(&reg, &i, "id_any"),
            vec![erased(erased("x"))]
        ))),
        "x"
    );

    let arr = erased(Arr::<V, ()>::new(vec![erased(1i64), erased(2i64)]));
    let boxed = call_sync(handler(&reg, &i, "boxed"), vec![arr]);
    // SAFETY: `boxed`'s glue erased its return from a `Boxed<T, Pure, Rt>`.
    let Boxed::<V, Pure, Tiny>(items, _) = unsafe { Boxed::materialize(&Tiny, boxed) };
    assert_eq!(items.len(), 2);
    let boxed = Boxed::<V, (), Tiny>(items, PhantomData).erase(&Tiny);

    let double = V::Closure(Closure(Arc::new(|args| {
        let [n] = <[V; 1]>::try_from(args)
            .unwrap_or_else(|args| panic!("double takes one argument, got {}", args.len()));
        erased(open::<i64>(n) * 2)
    })));
    let out = call_async(handler(&reg, &i, "apply"), vec![boxed, double]).await;
    // SAFETY: `apply`'s glue erased its return from a `Boxed<U, E, Rt>`.
    let Boxed::<V, (), Tiny>(items, _) = unsafe { Boxed::materialize(&Tiny, out) };
    let doubled: Vec<i64> = items.into_iter().map(open::<i64>).collect();
    assert_eq!(doubled, vec![2, 4]);

    // An object crosses as its fields (RFC-0032): the handler receives
    // `Obj<V>` and returns one.
    let point = erased(acvus_extern::Obj(
        [
            (Tiny.symbol("x"), erased(21i64)),
            (Tiny.symbol("label"), erased("p".to_owned())),
        ]
        .into_iter()
        .collect::<acvus_extern::FxHashMap<_, _>>(),
    ));
    let out = call_async(handler(&reg, &i, "fetch"), vec![point]).await;
    let acvus_extern::Obj(mut fields) = open::<acvus_extern::Obj<V>>(out);
    assert_eq!(open::<i64>(fields.remove(&Tiny.symbol("x")).unwrap()), 42);
    assert_eq!(
        open::<String>(fields.remove(&Tiny.symbol("label")).unwrap()),
        "p"
    );
    assert!(fields.is_empty());
}

#[test]
#[should_panic(expected = "materialize: value is not a i64")]
fn wrong_argument_type_panics_trusting_typeck() {
    let (i, reg) = combined::<Tiny>();
    let _ = call_sync(handler(&reg, &i, "add"), vec![erased("a"), erased(2i64)]);
}

#[test]
fn a_shared_signature_collects_its_instances_and_bounds_what_requires_it() {
    let (i, reg) = combined::<TypesOnly>();
    let point = acvus_extern::Ty::Object(
        [
            (i.intern("x"), acvus_extern::Ty::I64),
            (i.intern("label"), acvus_extern::Ty::String),
        ]
        .into_iter()
        .collect(),
    );
    let eq_fn = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(&i, "eq"))
        .expect("eq");
    let acvus_extern::FnKind::Extern { bounds, .. } = &eq_fn.kind else {
        panic!("eq is extern")
    };
    assert_eq!(
        bounds[0],
        acvus_extern::TyVarBound::OneOf(vec![
            acvus_extern::PolyTy::I64,
            acvus_extern::lift_to_poly(&point)
        ])
    );
    let same = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(&i, "same"))
        .expect("same");
    let acvus_extern::FnKind::Extern { bounds, .. } = &same.kind else {
        panic!("same is extern")
    };
    assert_eq!(
        bounds[0],
        acvus_extern::TyVarBound::OneOf(vec![
            acvus_extern::PolyTy::I64,
            acvus_extern::lift_to_poly(&point)
        ])
    );
    assert_eq!(reg.handlers[&qref(&i, "eq")].len(), 2);
}

#[test]
fn a_state_parameter_is_held_by_the_handler() {
    let (i, reg) = combined::<Tiny>();
    let out = call_sync(handler(&reg, &i, "greet"), vec![erased("bob".to_string())]);
    assert_eq!(open::<String>(out), "hello, bob");
}

#[test]
fn a_second_instance_for_one_type_is_refused() {
    let i = Interner::new();
    let err = Externs::combine(vec![registry::<TypesOnly>(), registry::<TypesOnly>()], &i)
        .err()
        .expect("two registries declare the same names");
    assert!(matches!(err, acvus_extern::CombineError::DuplicateName(_)));
}

#[test]
fn stand_ins_name_their_positions() {
    let i = Interner::new();
    let vars = acvus_extern::PolyVars::fresh(acvus_extern::VarCounts {
        tys: 2,
        effects: 1,
        lens: 1,
        identities: 0,
    });
    assert_eq!(
        <acvus_extern::Typeck<1> as TyArg>::poly_ty(&i, &vars),
        PolyTy::Var(1)
    );
    assert_eq!(
        <Eff<0> as acvus_extern::EffectArg>::poly_effect(&vars),
        EffectTerm::Var(0)
    );
    assert_eq!(
        <Arr<acvus_extern::Typeck<0>, acvus_extern::Len<0>> as TyArg>::poly_ty(&i, &vars),
        PolyTy::Array(Box::new(PolyTy::Var(0)), LenTerm::Var(0))
    );
}

#[derive(TyArg, Debug, PartialEq)]
enum Shape {
    Dot,
    Circle(i64),
    Rect { w: i64, h: i64 },
}

#[test]
fn a_derived_enum_is_the_language_s_enum_of_the_same_name() {
    let i = Interner::new();
    let vars = acvus_extern::PolyVars::fresh(acvus_extern::VarCounts::default());
    let rect = PolyTy::Object(
        [(i.intern("w"), PolyTy::I64), (i.intern("h"), PolyTy::I64)]
            .into_iter()
            .collect(),
    );
    assert_eq!(
        <Shape as TyArg>::poly_ty(&i, &vars),
        PolyTy::Enum {
            name: i.intern("Shape"),
            variants: [
                (i.intern("Dot"), None),
                (i.intern("Circle"), Some(Box::new(PolyTy::I64))),
                (i.intern("Rect"), Some(Box::new(rect))),
            ]
            .into_iter()
            .collect(),
        }
    );
}

#[test]
fn a_result_is_the_language_s_result_of_its_two_types() {
    let i = Interner::new();
    let vars = acvus_extern::PolyVars::fresh(acvus_extern::VarCounts::default());
    assert_eq!(
        <Result<i64, String> as TyArg>::poly_ty(&i, &vars),
        PolyTy::Result(Box::new(PolyTy::I64), Box::new(PolyTy::String))
    );
}

// -- Monomorphize -------------------------------------------------------

/// Twice the value, as each member type defines it.
trait Twice {
    fn twice(self) -> Self;
}

impl Twice for i64 {
    fn twice(self) -> Self {
        self * 2
    }
}

impl Twice for String {
    fn twice(self) -> Self {
        format!("{self}{self}")
    }
}

#[extern_fn(effect = pure)]
fn double<A>(a: A) -> A
where
    A: acvus_extern::Monomorphize<(i64, String)> + Twice,
{
    a.twice()
}

#[extern_fn(effect = pure)]
fn first_or<A>(v: Option<A>, fallback: A) -> A
where
    A: acvus_extern::Monomorphize<(i64, String)>,
{
    v.unwrap_or(fallback)
}

/// The member type appears only inside an extension type here.
#[extern_fn(effect = pure)]
fn box_count<A, Rt>(v: Boxed<A, Pure, Rt>) -> i64
where
    A: acvus_extern::Monomorphize<(i64, String)>,
    Rt: Runtime,
{
    v.0.len() as i64
}

fn mono_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        fns: [double, first_or, box_count],
    }
}

#[test]
fn a_monomorphized_parameter_declares_its_members_as_the_bound() {
    let i = Interner::new();
    let reg = Externs::combine(vec![mono_registry::<TypesOnly>()], &i).expect("registries combine");
    let double = reg
        .functions
        .iter()
        .find(|f| i.resolve(f.qref.name) == "double")
        .expect("double");
    let acvus_extern::FnKind::Extern { bounds, .. } = &double.kind else {
        panic!("double is extern");
    };
    assert_eq!(
        bounds,
        &vec![acvus_extern::TyVarBound::OneOf(vec![
            acvus_extern::PolyTy::I64,
            acvus_extern::PolyTy::String
        ])]
    );
    let shape = fn_ty(&double.ty);
    assert_eq!(shape.params, vec![PolyTy::Var(0)]);
    assert_eq!(shape.ret, PolyTy::Var(0));
}

fn call_type(
    params: Vec<acvus_extern::Ty>,
    ret: acvus_extern::Ty,
    i: &Interner,
) -> acvus_extern::Ty {
    acvus_extern::Ty::Fn {
        params: params
            .into_iter()
            .enumerate()
            .map(|(k, ty)| acvus_extern::ParamTerm::new(i.intern(&format!("_{k}")), ty))
            .collect(),
        ret: Box::new(ret),
        captures: vec![],
        effect: EffectTerm::Known(Effect::PURE),
    }
}

#[test]
fn the_instances_the_compiler_sees_are_the_handlers_in_that_order() {
    let i = Interner::new();
    let reg = Externs::combine(vec![mono_registry::<Tiny>()], &i).expect("registries combine");

    let on_int = call_type(vec![acvus_extern::Ty::I64], acvus_extern::Ty::I64, &i);
    let h = instance_for(&reg, &i, "double", &on_int).unwrap();
    assert_eq!(open::<i64>(call_sync(h, vec![erased(21i64)])), 42);

    let on_str = call_type(vec![acvus_extern::Ty::String], acvus_extern::Ty::String, &i);
    let h = instance_for(&reg, &i, "double", &on_str).unwrap();
    assert_eq!(
        open::<String>(call_sync(h, vec![erased(String::from("ab"))])),
        "abab"
    );

    let on_float = call_type(vec![acvus_extern::Ty::Float], acvus_extern::Ty::Float, &i);
    assert!(instance_for(&reg, &i, "double", &on_float).is_none());

    let ty = call_type(
        vec![
            acvus_extern::Ty::Option(Box::new(acvus_extern::Ty::String)),
            acvus_extern::Ty::String,
        ],
        acvus_extern::Ty::String,
        &i,
    );
    let h = instance_for(&reg, &i, "first_or", &ty).unwrap();
    assert_eq!(
        open::<String>(call_sync(
            h,
            vec![erased(Option::<V>::None), erased(String::from("x"))]
        )),
        "x"
    );

    let boxed_of = |t: acvus_extern::Ty| acvus_extern::Ty::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Box")),
        type_args: vec![TypeArg::uniform(t)],
        effect_args: vec![EffectTerm::Known(Effect::PURE)],
        identity_args: vec![],
    };
    let ty = call_type(
        vec![boxed_of(acvus_extern::Ty::String)],
        acvus_extern::Ty::I64,
        &i,
    );
    let h = instance_for(&reg, &i, "box_count", &ty).unwrap();
    let payload = Boxed::<String, Pure, Tiny>(
        vec![erased(String::from("a")), erased(String::from("b"))],
        PhantomData,
    )
    .erase(&Tiny);
    assert_eq!(open::<i64>(call_sync(h, vec![payload])), 2);
    let ty = call_type(
        vec![boxed_of(acvus_extern::Ty::Float)],
        acvus_extern::Ty::I64,
        &i,
    );
    let fallback = instance_for(&reg, &i, "box_count", &ty).unwrap();
    let payload = Boxed::<V, Pure, Tiny>(
        vec![erased(1.5f64), erased(2.5f64), erased(3.5f64)],
        PhantomData,
    )
    .erase(&Tiny);
    assert_eq!(open::<i64>(call_sync(fallback, vec![payload])), 3);
}

// -- Polymorphic instances (RFC-0027) ---------------------------------

extern_signature! { ns: "t", fn first<C, T>(c: C) -> T where C: TyVar, T: TyVar; }

#[extern_fn(instance_of = first, effect = pure)]
fn first_arr<T, N>(a: Arr<T, N>) -> T
where
    T: TyVar,
    N: LenVar,
{
    a.0.into_iter().next().expect("first: empty array")
}

#[extern_fn(instance_of = first, effect = pure)]
fn first_opt<T>(v: Option<T>) -> T
where
    T: TyVar,
{
    v.expect("first: none")
}

#[extern_fn(instance_of = first, effect = pure)]
fn first_arr_again<T, N>(a: Arr<T, N>) -> T
where
    T: TyVar,
    N: LenVar,
{
    first_arr(a)
}

fn first_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        signatures: [first],
        fns: [first_arr, first_opt],
    }
}

fn overlapping_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t2",
        fns: [first_arr_again],
    }
}

#[test]
fn a_polymorphic_instance_is_selected_by_the_argument_s_shape() {
    let i = Interner::new();
    let reg = Externs::combine(vec![first_registry::<Tiny>()], &i).expect("registries combine");
    let first_fn = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(&i, "first"))
        .expect("first");
    let acvus_extern::FnKind::Extern { bounds, instances } = &first_fn.kind else {
        panic!("first is extern")
    };
    assert_eq!(instances.concrete.len(), 2);
    assert!(!instances.generic);
    let acvus_extern::TyVarBound::OneOf(shapes) = &bounds[0] else {
        panic!("the instance variable is bounded")
    };
    assert!(
        shapes.iter().any(|s| matches!(s, PolyTy::Array(..))),
        "{shapes:?}"
    );
    assert!(
        shapes.iter().any(|s| matches!(s, PolyTy::Option(..))),
        "{shapes:?}"
    );

    let on_array = call_type(
        vec![acvus_extern::Ty::Array(
            Box::new(acvus_extern::Ty::I64),
            acvus_extern::LenTerm::Known(2),
        )],
        acvus_extern::Ty::I64,
        &i,
    );
    let arr = erased(Arr::<V, ()>::new(vec![erased(7i64), erased(8i64)]));
    let h = instance_for(&reg, &i, "first", &on_array).unwrap();
    assert_eq!(open::<i64>(call_sync(h, vec![arr])), 7);

    let on_option = call_type(
        vec![acvus_extern::Ty::Option(Box::new(acvus_extern::Ty::String))],
        acvus_extern::Ty::String,
        &i,
    );
    let h = instance_for(&reg, &i, "first", &on_option).unwrap();
    assert_eq!(
        open::<String>(call_sync(h, vec![erased(Some(erased(String::from("s"))))])),
        "s"
    );

    let on_int = call_type(vec![acvus_extern::Ty::I64], acvus_extern::Ty::I64, &i);
    assert!(instance_for(&reg, &i, "first", &on_int).is_none());
}

#[test]
fn two_instances_whose_types_unify_are_refused() {
    let i = Interner::new();
    let err = Externs::combine(
        vec![first_registry::<Tiny>(), overlapping_registry::<Tiny>()],
        &i,
    )
    .err()
    .expect("the second Array instance is refused");
    assert!(
        matches!(err, acvus_extern::CombineError::DuplicateInstance { .. }),
        "{err:?}"
    );
}

/// An `Inline` element is read, copied and edited through `Erased` with no
/// runtime in hand; a `String` still needs one.
#[test]
fn erased_inline_derefs_without_a_runtime() {
    use acvus_extern::Erased;
    let rt = Tiny;
    let mut n: Erased<Tiny, i64> = Erased::new(&rt, 41);
    assert_eq!(*n, 41);
    assert_eq!(n.get(), 41);
    *n += 1;
    assert_eq!(n, Erased::new(&rt, 42));
    assert_eq!(n.into_inner(&rt), 42);
    let s: Erased<Tiny, String> = Erased::new(&rt, "a".to_string());
    assert_eq!(s.as_ref(&rt), "a");
}
