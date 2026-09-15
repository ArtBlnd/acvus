//! RFC-0009 at its contract: one Rust signature yields the acvus type and
//! a handler for any runtime. `Tiny` is a runtime written for this test
//! alone, so nothing here depends on an interpreter.

use std::any::Any;
use std::future::Ready;
use std::marker::PhantomData;
use std::sync::Arc;

use acvus_extern::{
    Arr, CallToken, ClosureFn, Eff, Effect, EffectTerm, EffectVar, ExternError, ExternFn,
    ExternHandler, ExternItems,
    ExternRegistry, ExternType, Fn1, Interner, LenTerm, LenVar, PolyTy, Pure, Runtime, TyArg,
    TyVar, TypeRegistry, TypesOnly, extern_fn, extern_registry,
};

// -- A runtime for this test ------------------------------------------

/// A value is either a Rust value boxed whole, or a closure. `Erased` never
/// compares equal: the test runtime has no view into what it holds.
#[derive(Debug)]
enum V {
    Erased(Box<dyn Any + Send + Sync>),
    Closure(Closure),
}

impl PartialEq for V {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (V::Closure(a), V::Closure(b)) => Arc::ptr_eq(&a.0, &b.0),
            _ => false,
        }
    }
}

#[derive(Clone)]
struct Closure(Arc<dyn Fn(&[&V]) -> V + Send + Sync>);

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
        V::Closure(_) => panic!("peek: value is a closure, not a {}", std::any::type_name::<T>()),
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
    }
}

#[derive(Clone)]
struct Tiny;

impl Tiny {
    fn call(&self, f: &V, args: &[&V]) -> Result<V, ExternError> {
        match f {
            V::Closure(c) => Ok((c.0)(args)),
            V::Erased(_) => Err(ExternError::internal(
                "call on a value that is not a closure",
            )),
        }
    }
}

impl Runtime for Tiny {
    type Value = V;
    type Error = ExternError;
    type CallFuture<'a> = Ready<Result<V, ExternError>>;

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
    fn call_0<'a>(&'a self, f: &'a V, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, &[]))
    }
    fn call_1<'a>(&'a self, f: &'a V, a: &'a V, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, &[a]))
    }
    fn call_n<'a>(&'a self, f: &'a V, args: &[&'a V], _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, args))
    }
}

// -- Declarations under test ------------------------------------------

#[derive(ExternType)]
#[extern_type(name = "Box")]
struct Boxed<T, E, Rt>(Vec<Rt::Value>, PhantomData<(T, E)>)
where
    T: TyVar,
    E: EffectVar,
    Rt: Runtime;

/// A value that is a source of its own: it carries an identity, so it moves.
#[derive(ExternType)]
struct Token<I>(i64, PhantomData<I>)
where
    I: acvus_extern::IdentityVar;

#[derive(TyArg, Debug, PartialEq)]
struct Point {
    x: i64,
    label: String,
}

#[extern_fn(effect = pure)]
fn add<R>(_: &R, a: i64, b: i64) -> i64
where
    R: Runtime,
{
    a + b
}

#[extern_fn(name = "id_any", effect = E)]
fn identity<T, E, R>(_: &R, v: T) -> T
where
    T: TyVar,
    E: EffectVar,
    R: Runtime,
{
    v
}

#[extern_fn(effect = pure)]
async fn apply<T, U, E, Rt>(
    rt: &Rt,
    v: Boxed<T, E, Rt>,
    f: Fn1<T, U, E, Rt>,
) -> Result<Boxed<U, E, Rt>, Rt::Error>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    let mut out = Vec::with_capacity(v.0.len());
    for item in &v.0 {
        out.push(f.call(rt, (item,)).await?);
    }
    Ok(Boxed(out, PhantomData))
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
async fn fetch<R>(_: &R, p: Point) -> Result<Point, ExternError>
where
    R: Runtime,
{
    Ok(Point {
        x: p.x * 2,
        label: p.label,
    })
}

#[extern_fn(effect = idempotent)]
fn take_token<I, R>(_: &R, t: Token<I>) -> i64
where
    I: acvus_extern::IdentityVar,
    R: Runtime,
{
    t.0
}

/// A fresh draw: two draws in either order are the same program.
#[extern_fn(effect = idempotent, commutative)]
fn draw<R>(_: &R) -> i64
where
    R: Runtime,
{
    4
}

/// Adds `by` to the lent place and returns the new value.
#[extern_fn(effect = pure)]
fn bump<R>(_: &R, n: &mut i64, by: i64) -> i64
where
    R: Runtime,
{
    *n += by;
    *n
}

fn registry<R>() -> ExternRegistry<R>
where
    R: Runtime,
{
    extern_registry! {
        types: [Boxed<_, _, R>, Token<_>],
        fns: [add, identity, apply, boxed, fetch, take_token, draw, bump],
    }
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
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry::<TypesOnly>().register(&i, &mut tr);
    let add = fn_ty(find(&reg.functions, &i, "add"));
    assert_eq!(add.params, vec![PolyTy::Int, PolyTy::Int]);
    assert_eq!(add.ret, PolyTy::Int);
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
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry::<TypesOnly>().register(&i, &mut tr);

    let id = fn_ty(find(&reg.functions, &i, "id_any"));
    assert_eq!(id.params, vec![PolyTy::Var(0)]);
    assert_eq!(id.ret, PolyTy::Var(0));
    assert_eq!(id.effect, EffectTerm::Var(0));

    let apply = fn_ty(find(&reg.functions, &i, "apply"));
    let boxed_of = |t: PolyTy| PolyTy::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Box")),
        type_args: vec![t],
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
            type_args: vec![PolyTy::Var(0)],
            effect_args: vec![EffectTerm::Known(Effect::PURE)],
            identity_args: vec![],
        }
    );
}

#[test]
fn types_and_casts_reach_the_type_registry() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    registry::<TypesOnly>().register(&i, &mut tr);
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

fn call_sync(handler: &ExternHandler<Tiny>, args: Vec<V>) -> Result<V, ExternError> {
    call_sync_lending(handler, args).map(|r| r.value)
}

fn call_sync_lending(
    handler: &ExternHandler<Tiny>,
    args: Vec<V>,
) -> Result<acvus_extern::Returned<V>, ExternError> {
    match handler {
        ExternHandler::Sync(f) => f(&Tiny, args),
        ExternHandler::Async(_) => panic!("expected a sync handler"),
    }
}

#[test]
fn a_borrowed_parameter_is_declared_and_given_back() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry::<Tiny>().register(&i, &mut tr);
    let PolyTy::Fn { params, .. } = find(&reg.functions, &i, "bump") else {
        panic!("bump is a function");
    };
    assert_eq!(params[0].mode, acvus_extern::ParamMode::BorrowMut);
    assert_eq!(params[1].mode, acvus_extern::ParamMode::Value);

    let h = handler(&reg, &i, "bump");
    let r = call_sync_lending(h, vec![erased(40i64), erased(2i64)]).unwrap();
    assert_eq!(open::<i64>(r.value), 42);
    let [lent] = <[V; 1]>::try_from(r.lent).expect("one lent place");
    assert_eq!(open::<i64>(lent), 42, "the lent place comes back changed");
}

async fn call_async(handler: &ExternHandler<Tiny>, args: Vec<V>) -> Result<V, ExternError> {
    match handler {
        ExternHandler::Async(f) => f(Tiny, args).await.map(|r| r.value),
        ExternHandler::Sync(_) => panic!("expected an async handler"),
    }
}

fn handler<'a>(
    reg: &'a acvus_extern::Registered<Tiny>,
    i: &Interner,
    name: &str,
) -> &'a ExternHandler<Tiny> {
    match &reg.handlers[&acvus_extern::QualifiedRef::root(i.intern(name))] {
        acvus_extern::ExternEntry::Single(h) => h,
        acvus_extern::ExternEntry::Mono(_) => {
            panic!("{name} is monomorphized; select by call type")
        }
    }
}

#[tokio::test]
async fn handlers_run_the_rust_body_on_the_test_runtime() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry::<Tiny>().register(&i, &mut tr);

    assert_eq!(
        open::<i64>(
            call_sync(handler(&reg, &i, "add"), vec![erased(40i64), erased(2i64)]).unwrap()
        ),
        42
    );
    assert_eq!(
        open::<&str>(open::<V>(
            call_sync(handler(&reg, &i, "id_any"), vec![erased(erased("x"))]).unwrap()
        )),
        "x"
    );

    let arr = erased(Arr::<V, ()>::new(vec![erased(1i64), erased(2i64)]));
    let boxed = call_sync(handler(&reg, &i, "boxed"), vec![arr]).unwrap();
    let Boxed::<V, Pure, Tiny>(items, _) = open(boxed) else {
        unreachable!("boxed returns a Box")
    };
    assert_eq!(items.len(), 2);
    let boxed = erased(Boxed::<V, (), Tiny>(items, PhantomData));

    let double = V::Closure(Closure(Arc::new(|args| {
        let [n] = args else {
            panic!("double takes one argument, got {}", args.len())
        };
        erased(peek::<i64>(n) * 2)
    })));
    let out = call_async(handler(&reg, &i, "apply"), vec![boxed, double])
        .await
        .unwrap();
    let Boxed::<V, (), Tiny>(items, _) = open(out);
    let doubled: Vec<i64> = items.into_iter().map(open::<i64>).collect();
    assert_eq!(doubled, vec![2, 4]);

    let point = erased(Point {
        x: 21,
        label: "p".to_owned(),
    });
    let out = call_async(handler(&reg, &i, "fetch"), vec![point])
        .await
        .unwrap();
    assert_eq!(
        open::<Point>(out),
        Point {
            x: 42,
            label: "p".to_owned()
        }
    );
}

#[test]
#[should_panic(expected = "materialize: value is not a i64")]
fn wrong_argument_type_panics_trusting_typeck() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry::<Tiny>().register(&i, &mut tr);
    let _ = call_sync(handler(&reg, &i, "add"), vec![erased("a"), erased(2i64)]);
}

#[test]
fn closure_declaration_takes_its_type_from_the_closure() {
    let i = Interner::new();
    let registry = ExternRegistry::<TypesOnly>::new(|i| ExternItems {
        types: vec![],
        fns: vec![
            ExternFn::sync(i, "shout", |_: &TypesOnly, s: String| Ok(s.to_uppercase()))
                .with_effect(Effect::PURE),
        ],
    });
    let mut tr = TypeRegistry::new();
    let reg = registry.register(&i, &mut tr);
    let shout = fn_ty(find(&reg.functions, &i, "shout"));
    assert_eq!(shout.params, vec![PolyTy::String]);
    assert_eq!(shout.ret, PolyTy::String);
    assert_eq!(shout.effect, EffectTerm::Known(Effect::PURE));
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
fn double<A, R>(_: &R, a: A) -> A
where
    A: acvus_extern::Monomorphize<(i64, String)> + Twice,
    R: Runtime,
{
    a.twice()
}

#[extern_fn(effect = pure)]
fn first_or<A, R>(_: &R, v: Option<A>, fallback: A) -> A
where
    A: acvus_extern::Monomorphize<(i64, String)>,
    R: Runtime,
{
    v.unwrap_or(fallback)
}

/// The member type appears only inside an extension type here.
#[extern_fn(effect = pure)]
fn box_count<A, Rt>(_: &Rt, v: Boxed<A, Pure, Rt>) -> i64
where
    A: acvus_extern::Monomorphize<(i64, String)>,
    Rt: Runtime,
{
    v.0.len() as i64
}

fn mono_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        fns: [double, first_or, box_count],
    }
}

#[test]
fn a_monomorphized_parameter_declares_its_members_as_the_bound() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = mono_registry::<TypesOnly>().register(&i, &mut tr);
    let double = reg
        .functions
        .iter()
        .find(|f| i.resolve(f.qref.name) == "double")
        .expect("double");
    let acvus_extern::FnKind::Extern { bounds } = &double.kind else {
        panic!("double is extern");
    };
    assert_eq!(
        bounds,
        &vec![acvus_extern::TyVarBound::OneOf(vec![
            acvus_extern::Ty::Int,
            acvus_extern::Ty::String
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
fn the_call_type_selects_the_instance() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = mono_registry::<Tiny>().register(&i, &mut tr);
    let entry = &reg.handlers[&acvus_extern::QualifiedRef::root(i.intern("double"))];

    let on_int = call_type(vec![acvus_extern::Ty::Int], acvus_extern::Ty::Int, &i);
    let h = entry.select(&on_int).unwrap();
    assert_eq!(open::<i64>(call_sync(h, vec![erased(21i64)]).unwrap()), 42);

    let on_str = call_type(vec![acvus_extern::Ty::String], acvus_extern::Ty::String, &i);
    let h = entry.select(&on_str).unwrap();
    assert_eq!(
        open::<String>(call_sync(h, vec![erased(String::from("ab"))]).unwrap()),
        "abab"
    );

    let on_float = call_type(vec![acvus_extern::Ty::Float], acvus_extern::Ty::Float, &i);
    assert!(entry.select(&on_float).is_err());

    let nested = &reg.handlers[&acvus_extern::QualifiedRef::root(i.intern("first_or"))];
    let ty = call_type(
        vec![
            acvus_extern::Ty::Option(Box::new(acvus_extern::Ty::String)),
            acvus_extern::Ty::String,
        ],
        acvus_extern::Ty::String,
        &i,
    );
    let h = nested.select(&ty).unwrap();
    assert_eq!(
        open::<String>(
            call_sync(
                h,
                vec![erased(Option::<String>::None), erased(String::from("x"))]
            )
            .unwrap()
        ),
        "x"
    );

    let inside = &reg.handlers[&acvus_extern::QualifiedRef::root(i.intern("box_count"))];
    let boxed_of = |t: acvus_extern::Ty| acvus_extern::Ty::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Box")),
        type_args: vec![t],
        effect_args: vec![EffectTerm::Known(Effect::PURE)],
        identity_args: vec![],
    };
    let ty = call_type(
        vec![boxed_of(acvus_extern::Ty::String)],
        acvus_extern::Ty::Int,
        &i,
    );
    let h = inside.select(&ty).unwrap();
    let payload = erased(Boxed::<String, Pure, Tiny>(
        vec![erased(String::from("a")), erased(String::from("b"))],
        PhantomData,
    ));
    assert_eq!(open::<i64>(call_sync(h, vec![payload]).unwrap()), 2);
    let ty = call_type(
        vec![boxed_of(acvus_extern::Ty::Float)],
        acvus_extern::Ty::Int,
        &i,
    );
    let fallback = inside.select(&ty).unwrap();
    let payload = erased(Boxed::<V, Pure, Tiny>(
        vec![erased(1.5f64), erased(2.5f64), erased(3.5f64)],
        PhantomData,
    ));
    assert_eq!(open::<i64>(call_sync(fallback, vec![payload]).unwrap()), 3);
}
