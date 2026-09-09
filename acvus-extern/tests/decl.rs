//! RFC-0009 at its contract: one Rust signature yields the acvus type and
//! the runtime handler.

use acvus_extern::{
    Arr, Eff, Effect, EffectTerm, EffectVar, ExternFn, ExternHandler, ExternItems, ExternRegistry,
    ExternType, Fn1, Interner, IntoValue, LenTerm, LenVar, PolyTy, Pure, RuntimeError, TyArg,
    TyVar, TypeRegistry, Value, extern_fn, extern_registry,
};
use std::marker::PhantomData;

#[derive(ExternType)]
#[extern_type(name = "Box")]
struct Boxed<T, E>(Vec<Value>, PhantomData<(T, E)>)
where
    T: TyVar,
    E: EffectVar;

#[derive(ExternType)]
#[extern_type(move_only)]
struct Token(i64);

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
fn apply<T, U, E>(_: &Interner, v: Boxed<T, E>, f: Fn1<T, U, E>) -> Boxed<U, E>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
{
    let _ = f;
    Boxed(v.0, PhantomData)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn boxed<T, N>(i: &Interner, items: Arr<T, N>) -> Boxed<T, Pure>
where
    T: TyVar,
    N: LenVar,
{
    Boxed(
        items.0.into_iter().map(|v| v.into_value(i)).collect(),
        PhantomData,
    )
}

#[extern_fn]
async fn fetch(_: Interner, p: Point) -> Result<Point, RuntimeError> {
    Ok(Point {
        x: p.x * 2,
        label: p.label,
    })
}

#[extern_fn(effect = idempotent)]
fn take_token(_: &Interner, t: Token) -> i64 {
    t.0
}

fn registry() -> ExternRegistry {
    extern_registry! {
        types: [Boxed<_, _>, Token],
        fns: [add, identity, apply, boxed, fetch, take_token],
    }
}

fn fn_ty(ty: &PolyTy) -> (Vec<PolyTy>, PolyTy, EffectTerm<acvus_extern::Poly>) {
    let PolyTy::Fn {
        params,
        ret,
        effect,
        ..
    } = ty
    else {
        panic!("not a function type: {ty:?}");
    };
    (
        params.iter().map(|p| p.ty.clone()).collect(),
        (**ret).clone(),
        *effect,
    )
}

fn find<'a>(reg: &'a [acvus_mir::graph::Function], i: &Interner, name: &str) -> &'a PolyTy {
    &reg.iter()
        .find(|f| i.resolve(f.qref.name) == name)
        .expect(name)
        .ty
}

#[test]
fn concrete_signature_and_declared_effect() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry().register(&i, &mut tr);
    let (params, ret, effect) = fn_ty(find(&reg.functions, &i, "add"));
    assert_eq!(params, vec![PolyTy::Int, PolyTy::Int]);
    assert_eq!(ret, PolyTy::Int);
    assert_eq!(effect, EffectTerm::Known(Effect::Pure));

    let (_, _, effect) = fn_ty(find(&reg.functions, &i, "fetch"));
    assert_eq!(effect, EffectTerm::Known(Effect::Opaque));
    let (_, _, effect) = fn_ty(find(&reg.functions, &i, "take_token"));
    assert_eq!(effect, EffectTerm::Known(Effect::Idempotent));
}

#[test]
fn generic_parameters_become_positional_variables() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry().register(&i, &mut tr);

    let (params, ret, effect) = fn_ty(find(&reg.functions, &i, "id_any"));
    assert_eq!(params, vec![PolyTy::Var(0)]);
    assert_eq!(ret, PolyTy::Var(0));
    assert_eq!(effect, EffectTerm::Var(0));

    let (params, ret, _) = fn_ty(find(&reg.functions, &i, "apply"));
    let boxed_of = |t: PolyTy| PolyTy::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Box")),
        type_args: vec![t],
        effect_args: vec![EffectTerm::Var(0)],
    };
    assert_eq!(params[0], boxed_of(PolyTy::Var(0)));
    let PolyTy::Fn {
        params: fparams,
        ret: fret,
        effect,
        ..
    } = &params[1]
    else {
        panic!("callback is a function type");
    };
    assert_eq!(fparams[0].ty, PolyTy::Var(0));
    assert_eq!(**fret, PolyTy::Var(1));
    assert_eq!(*effect, EffectTerm::Var(0));
    assert_eq!(ret, boxed_of(PolyTy::Var(1)));

    let (params, ret, _) = fn_ty(find(&reg.functions, &i, "boxed"));
    assert_eq!(
        params,
        vec![PolyTy::Array(Box::new(PolyTy::Var(0)), LenTerm::Var(0))]
    );
    assert_eq!(
        ret,
        PolyTy::UserDefined {
            id: acvus_extern::QualifiedRef::root(i.intern("Box")),
            type_args: vec![PolyTy::Var(0)],
            effect_args: vec![EffectTerm::Known(Effect::Pure)],
        }
    );
}

#[test]
fn types_and_casts_reach_the_type_registry() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    registry().register(&i, &mut tr);
    let decl = tr.get(acvus_extern::QualifiedRef::root(i.intern("Box")));
    assert_eq!(decl.type_params.len(), 1);
    assert_eq!(decl.effect_params, 1);
    assert_eq!(
        tr.get(acvus_extern::QualifiedRef::root(i.intern("Token")))
            .type_params
            .len(),
        0
    );

    let rules = tr.rules_to(acvus_extern::QualifiedRef::root(i.intern("Box")));
    assert_eq!(rules.len(), 1);
    assert_eq!(i.resolve(rules[0].fn_ref.name), "boxed");
    assert_eq!(
        rules[0].from,
        PolyTy::Array(Box::new(PolyTy::Var(0)), LenTerm::Var(0))
    );
}

fn call_sync(
    handler: &ExternHandler,
    args: Vec<Value>,
    i: &Interner,
) -> Result<Value, RuntimeError> {
    match handler {
        ExternHandler::Sync(f) => f(args, i),
        ExternHandler::Async(_) => panic!("expected a sync handler"),
    }
}

async fn call_async(
    handler: &ExternHandler,
    args: Vec<Value>,
    i: &Interner,
) -> Result<Value, RuntimeError> {
    match handler {
        ExternHandler::Async(f) => f(args, i.clone()).await,
        ExternHandler::Sync(_) => panic!("expected an async handler"),
    }
}

fn handler<'a>(reg: &'a acvus_extern::Registered, i: &Interner, name: &str) -> &'a ExternHandler {
    let qref = acvus_extern::QualifiedRef::root(i.intern(name));
    match &reg.executables[&qref] {
        acvus_interpreter::Executable::Extern(h) => h,
        _ => panic!("{name} is an extern"),
    }
}

#[tokio::test]
async fn handlers_run_the_rust_body() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry().register(&i, &mut tr);

    assert_eq!(
        call_sync(
            handler(&reg, &i, "add"),
            vec![Value::Int(40), Value::Int(2)],
            &i
        )
        .unwrap(),
        Value::Int(42)
    );
    assert_eq!(
        call_sync(handler(&reg, &i, "id_any"), vec![Value::string("x")], &i).unwrap(),
        Value::string("x")
    );
    let arr = Value::array(vec![Value::Int(1), Value::Int(2)]);
    let out = call_sync(handler(&reg, &i, "boxed"), vec![arr], &i).unwrap();
    let Value::Extern(o) = out else {
        panic!("boxed returns an extension value")
    };
    assert_eq!(o.type_name.name, "Box");

    let point = Value::object(
        [
            (i.intern("x"), Value::Int(21)),
            (i.intern("label"), Value::string("p")),
        ]
        .into_iter()
        .collect(),
    );
    let out = call_async(handler(&reg, &i, "fetch"), vec![point], &i)
        .await
        .unwrap();
    assert_eq!(
        Point::from_value_for_test(out, &i),
        Point {
            x: 42,
            label: "p".to_owned()
        }
    );
}

impl Point {
    fn from_value_for_test(v: Value, i: &Interner) -> Self {
        <Point as acvus_extern::FromValue>::from_value(v, i).unwrap()
    }
}

#[test]
fn wrong_argument_type_is_a_runtime_error() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry().register(&i, &mut tr);
    let err = call_sync(
        handler(&reg, &i, "add"),
        vec![Value::string("a"), Value::Int(2)],
        &i,
    )
    .unwrap_err();
    assert!(err.to_string().contains("expected Int"), "{err}");
}

#[test]
fn move_only_value_rejects_a_shared_payload() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let reg = registry().register(&i, &mut tr);
    let token = Value::extern_value(acvus_extern::ExternValue::new(Token::TYPE_NAME, 5i64));
    let shared = token.clone();
    let err = call_sync(handler(&reg, &i, "take_token"), vec![token], &i).unwrap_err();
    assert!(err.to_string().contains("still shared"), "{err}");
    drop(shared);
}

#[test]
fn closure_declaration_takes_its_type_from_the_closure() {
    let i = Interner::new();
    let registry = ExternRegistry::new(|i| ExternItems {
        types: vec![],
        fns: vec![
            ExternFn::sync(i, "shout", |_: &Interner, s: String| Ok(s.to_uppercase()))
                .with_effect(Effect::Pure),
        ],
    });
    let mut tr = TypeRegistry::new();
    let reg = registry.register(&i, &mut tr);
    let (params, ret, effect) = fn_ty(find(&reg.functions, &i, "shout"));
    assert_eq!(params, vec![PolyTy::String]);
    assert_eq!(ret, PolyTy::String);
    assert_eq!(effect, EffectTerm::Known(Effect::Pure));
}

#[test]
fn stand_ins_name_their_positions() {
    let i = Interner::new();
    let vars = acvus_extern::PolyVars::fresh(2, 1, 1);
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
