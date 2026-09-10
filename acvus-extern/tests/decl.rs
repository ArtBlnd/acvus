//! RFC-0009 at its contract: one Rust signature yields the acvus type and
//! a handler for any runtime. `Tiny` is a runtime written for this test
//! alone, so nothing here depends on an interpreter.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_extern::{
    Arr, Astr, BoxFuture, Eff, Effect, EffectTerm, EffectVar, ExternError, ExternFn, ExternHandler,
    ExternItems, ExternRegistry, ExternType, ExternValue, Fn1, FromValue, FxHashMap, Interner,
    IntoValue, LenTerm, LenVar, PolyTy, Pure, Runtime, TyArg, TyVar, TypeRegistry, TypesOnly,
    extern_fn, extern_registry,
};

// -- A runtime for this test ------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum V {
    Unit,
    Int(i64),
    Str(String),
    Array(Vec<V>),
    Tuple(Vec<V>),
    Object(Vec<(Astr, V)>),
    Some(Box<V>),
    None,
    Extern(ExternValue),
    Closure(Closure),
}

#[derive(Clone)]
struct Closure(Arc<dyn Fn(Vec<V>) -> V + Send + Sync>);

impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "closure")
    }
}

impl PartialEq for Closure {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

struct Tiny;

fn wrong(shape: &str, got: &V) -> ExternError {
    ExternError::internal(format!("expected {shape}, got {got:?}"))
}

impl FromValue<Tiny> for V {
    fn from_value(value: V, _: &Interner) -> Result<Self, ExternError> {
        Ok(value)
    }
}

impl IntoValue<Tiny> for V {
    fn into_value(self, _: &Interner) -> V {
        self
    }
}

impl Runtime for Tiny {
    type Value = V;
    type Closure = Closure;
    type Error = ExternError;

    fn equals(a: &V, b: &V) -> bool {
        a == b
    }
    fn unit() -> V {
        V::Unit
    }
    fn into_unit(value: V) -> Result<(), ExternError> {
        match value {
            V::Unit => Ok(()),
            other => Err(wrong("Unit", &other)),
        }
    }
    fn int(n: i64) -> V {
        V::Int(n)
    }
    fn into_int(value: V) -> Result<i64, ExternError> {
        match value {
            V::Int(n) => Ok(n),
            other => Err(wrong("Int", &other)),
        }
    }
    fn float(_: f64) -> V {
        panic!("Tiny has no floats")
    }
    fn into_float(value: V) -> Result<f64, ExternError> {
        Err(wrong("Float", &value))
    }
    fn bool(_: bool) -> V {
        panic!("Tiny has no bools")
    }
    fn into_bool(value: V) -> Result<bool, ExternError> {
        Err(wrong("Bool", &value))
    }
    fn byte(_: u8) -> V {
        panic!("Tiny has no bytes")
    }
    fn into_byte(value: V) -> Result<u8, ExternError> {
        Err(wrong("Byte", &value))
    }
    fn string(s: String) -> V {
        V::Str(s)
    }
    fn into_string(value: V) -> Result<String, ExternError> {
        match value {
            V::Str(s) => Ok(s),
            other => Err(wrong("String", &other)),
        }
    }
    fn array(items: Vec<V>) -> V {
        V::Array(items)
    }
    fn into_array(value: V) -> Result<Vec<V>, ExternError> {
        match value {
            V::Array(items) => Ok(items),
            other => Err(wrong("Array", &other)),
        }
    }
    fn tuple(items: Vec<V>) -> V {
        V::Tuple(items)
    }
    fn into_tuple(value: V) -> Result<Vec<V>, ExternError> {
        match value {
            V::Tuple(items) => Ok(items),
            other => Err(wrong("Tuple", &other)),
        }
    }
    fn object(fields: FxHashMap<Astr, V>) -> V {
        V::Object(fields.into_iter().collect())
    }
    fn into_object(value: V) -> Result<FxHashMap<Astr, V>, ExternError> {
        match value {
            V::Object(fields) => Ok(fields.into_iter().collect()),
            other => Err(wrong("Object", &other)),
        }
    }
    fn some(_: &Interner, value: V) -> V {
        V::Some(Box::new(value))
    }
    fn none(_: &Interner) -> V {
        V::None
    }
    fn into_option(_: &Interner, value: V) -> Result<Option<V>, ExternError> {
        match value {
            V::Some(v) => Ok(Some(*v)),
            V::None => Ok(None),
            other => Err(wrong("Option", &other)),
        }
    }
    fn extern_value(value: ExternValue) -> V {
        V::Extern(value)
    }
    fn into_extern(value: V) -> Result<ExternValue, ExternError> {
        match value {
            V::Extern(o) => Ok(o),
            other => Err(wrong("Extern", &other)),
        }
    }
    fn closure(closure: Closure) -> V {
        V::Closure(closure)
    }
    fn into_closure(value: V) -> Result<Closure, ExternError> {
        match value {
            V::Closure(c) => Ok(c),
            other => Err(wrong("Closure", &other)),
        }
    }
    fn call(closure: &Closure, args: Vec<V>) -> BoxFuture<'_, Result<V, ExternError>> {
        Box::pin(std::future::ready(Ok((closure.0)(args))))
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
fn add(_: &Interner, a: i64, b: i64) -> i64 {
    a + b
}

#[extern_fn(name = "id_any", effect = E)]
fn identity<T, E>(_: &Interner, v: T) -> T
where
    T: TyVar,
    E: EffectVar,
{
    v
}

#[extern_fn(effect = pure)]
async fn apply<T, U, E, Rt>(
    i: Interner,
    v: Boxed<T, E, Rt>,
    f: Fn1<T, U, E, Rt>,
) -> Result<Boxed<U, E, Rt>, Rt::Error>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt>,
    U: TyVar + FromValue<Rt> + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    let mut out = Vec::with_capacity(v.0.len());
    for item in v.0 {
        let mapped: U = f.call(&i, T::from_value(item, &i)?).await?;
        out.push(mapped.into_value(&i));
    }
    Ok(Boxed(out, PhantomData))
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn boxed<T, N, Rt>(i: &Interner, items: Arr<T, N>) -> Boxed<T, Pure, Rt>
where
    T: TyVar + IntoValue<Rt>,
    N: LenVar,
    Rt: Runtime,
{
    Boxed(
        items.0.into_iter().map(|v| v.into_value(i)).collect(),
        PhantomData,
    )
}

#[extern_fn]
async fn fetch(_: Interner, p: Point) -> Result<Point, ExternError> {
    Ok(Point {
        x: p.x * 2,
        label: p.label,
    })
}

#[extern_fn(effect = idempotent)]
fn take_token<I>(_: &Interner, t: Token<I>) -> i64
where
    I: acvus_extern::IdentityVar,
{
    t.0
}

/// A fresh draw: two draws in either order are the same program.
#[extern_fn(effect = idempotent, commutative)]
fn draw(_: &Interner) -> i64 {
    4
}

fn registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        types: [Boxed<_, _, R>, Token<_>],
        fns: [add, identity, apply, boxed, fetch, take_token, draw],
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
        effect: *effect,
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

fn call_sync(handler: &ExternHandler<Tiny>, args: Vec<V>, i: &Interner) -> Result<V, ExternError> {
    match handler {
        ExternHandler::Sync(f) => f(args, i),
        ExternHandler::Async(_) => panic!("expected a sync handler"),
    }
}

async fn call_async(
    handler: &ExternHandler<Tiny>,
    args: Vec<V>,
    i: &Interner,
) -> Result<V, ExternError> {
    match handler {
        ExternHandler::Async(f) => f(args, i.clone()).await,
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
        call_sync(handler(&reg, &i, "add"), vec![V::Int(40), V::Int(2)], &i).unwrap(),
        V::Int(42)
    );
    assert_eq!(
        call_sync(handler(&reg, &i, "id_any"), vec![V::Str("x".into())], &i).unwrap(),
        V::Str("x".into())
    );

    let arr = V::Array(vec![V::Int(1), V::Int(2)]);
    let boxed = call_sync(handler(&reg, &i, "boxed"), vec![arr], &i).unwrap();
    let V::Extern(o) = &boxed else {
        panic!("boxed returns an extension value")
    };
    assert_eq!(o.type_name.name, "Box");

    let double = V::Closure(Closure(Arc::new(|args| match args.as_slice() {
        [V::Int(n)] => V::Int(n * 2),
        other => panic!("double got {other:?}"),
    })));
    let out = call_async(handler(&reg, &i, "apply"), vec![boxed, double], &i)
        .await
        .unwrap();
    let V::Extern(o) = out else {
        panic!("apply returns an extension value")
    };
    assert_eq!(
        o.downcast_ref::<Vec<V>>().expect("Box payload"),
        &vec![V::Int(2), V::Int(4)]
    );

    let point = V::Object(vec![
        (i.intern("x"), V::Int(21)),
        (i.intern("label"), V::Str("p".into())),
    ]);
    let out = call_async(handler(&reg, &i, "fetch"), vec![point], &i)
        .await
        .unwrap();
    assert_eq!(
        <Point as FromValue<Tiny>>::from_value(out, &i).unwrap(),
        Point {
            x: 42,
            label: "p".to_owned()
        }
    );
}

#[test]
fn wrong_argument_type_is_the_runtimes_error() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry::<Tiny>().register(&i, &mut tr);
    let err = call_sync(
        handler(&reg, &i, "add"),
        vec![V::Str("a".into()), V::Int(2)],
        &i,
    )
    .unwrap_err();
    assert!(err.to_string().contains("expected Int"), "{err}");
}

#[test]
fn move_only_value_rejects_a_shared_payload() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry::<Tiny>().register(&i, &mut tr);
    let token = V::Extern(ExternValue::new(Token::<()>::TYPE_NAME, 5i64));
    let shared = token.clone();
    let err = call_sync(handler(&reg, &i, "take_token"), vec![token], &i).unwrap_err();
    assert!(err.to_string().contains("still shared"), "{err}");
    drop(shared);
}

#[test]
fn closure_declaration_takes_its_type_from_the_closure() {
    let i = Interner::new();
    let registry = ExternRegistry::<TypesOnly>::new(|i| ExternItems {
        types: vec![],
        fns: vec![
            ExternFn::sync(i, "shout", |_: &Interner, s: String| Ok(s.to_uppercase()))
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
fn double<A>(_: &Interner, a: A) -> A
where
    A: acvus_extern::Monomorphize<(i64, String)> + Twice,
{
    a.twice()
}

#[extern_fn(effect = pure)]
fn first_or<A>(_: &Interner, v: Option<A>, fallback: A) -> A
where
    A: acvus_extern::Monomorphize<(i64, String)>,
{
    v.unwrap_or(fallback)
}

/// The member type appears only inside an extension type here.
#[extern_fn(effect = pure)]
fn box_count<A, Rt>(_: &Interner, v: Boxed<A, Pure, Rt>) -> i64
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
    assert_eq!(call_sync(h, vec![V::Int(21)], &i).unwrap(), V::Int(42));

    let on_str = call_type(vec![acvus_extern::Ty::String], acvus_extern::Ty::String, &i);
    let h = entry.select(&on_str).unwrap();
    assert_eq!(
        call_sync(h, vec![V::Str("ab".into())], &i).unwrap(),
        V::Str("abab".into())
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
        call_sync(h, vec![V::None, V::Str("x".into())], &i).unwrap(),
        V::Str("x".into())
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
    let payload = V::Extern(ExternValue::new(
        Boxed::<String, Pure, Tiny>::TYPE_NAME,
        vec![V::Str("a".into()), V::Str("b".into())],
    ));
    assert_eq!(call_sync(h, vec![payload], &i).unwrap(), V::Int(2));
    let ty = call_type(
        vec![boxed_of(acvus_extern::Ty::Float)],
        acvus_extern::Ty::Int,
        &i,
    );
    assert!(inside.select(&ty).is_err());
}
