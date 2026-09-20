//! RFC-0009 at its contract: one Rust signature yields the acvus type and
//! a handler for any runtime. `Tiny` is a runtime written for this test
//! alone, so nothing here depends on an interpreter.

use std::any::Any;
use std::future::Ready;
use std::marker::PhantomData;

use acvus_extern::{
    Arr, CallToken, ClosureFn, Effect, EffectTerm, Elements, ExternHandler, ExternType, Externs,
    Fn1, Interner, LenTerm, Nth, OneValue, Owned, PolyTy, Pure, Ref, Registry, Runtime, Slice,
    Task, TyArg, TypeArg, TypesOnly, Var, Words, extern_fn, extern_registry, extern_signature,
    kind,
};

// -- A runtime for this test ------------------------------------------

/// A value is a Rust value boxed whole, a closure, or a reference to
/// another value. `Erased` never compares equal: the test runtime has no
/// view into what it holds.
#[derive(Debug, Default, Clone, Copy)]
enum V {
    /// The value a handler took out of its argument slot.
    #[default]
    Taken,
    /// The language's `Option` (RFC-0022), held as a host pleases.
    None,
    /// RFC-0050 rule 8's two variant registers, held as a host pleases.
    Undef,
    Tag(acvus_extern::Astr),
    Some(*mut V),
    Erased(*mut (dyn Any + Send + Sync)),
    Closure(Closure),
    Reference(*const V),
}

// SAFETY: a `Reference` is used only while its target is live (RFC-0018).
unsafe impl Send for V {}
unsafe impl Sync for V {}

impl acvus_extern::Release for V {
    fn release(self) {
        match self {
            // SAFETY: an owning pointer comes from `Box::into_raw` and
            // reaches `release` once (RFC-0048).
            V::Some(payload) => (*unsafe { Box::from_raw(payload) }).release(),
            V::Erased(any) => drop(unsafe { Box::from_raw(any) }),
            V::Closure(c) => drop(unsafe { Box::from_raw(c.0) }),
            V::Taken | V::None | V::Undef | V::Tag(_) | V::Reference(_) => {}
        }
    }
}

impl PartialEq for V {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (V::Closure(a), V::Closure(b)) => std::ptr::addr_eq(a.0, b.0),
            (V::Reference(a), V::Reference(b)) => std::ptr::eq(*a, *b),
            _ => false,
        }
    }
}

#[derive(Clone, Copy)]
struct Closure(*mut (dyn Fn(Vec<V>) -> V + Send + Sync));

impl Closure {
    fn new(f: impl Fn(Vec<V>) -> V + Send + Sync + 'static) -> Self {
        Closure(Box::into_raw(Box::new(f)))
    }
}

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
    V::Erased(Box::into_raw(Box::new(value)))
}

/// A copy of the Rust value inside a lent `Erased`, or a panic naming the
/// mismatch.
fn peek<T>(value: &V) -> T
where
    T: Clone + Send + Sync + 'static,
{
    match value {
        V::Erased(any) => unsafe { &**any }
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
        V::None | V::Some(_) => panic!(
            "peek: value is an option, not a {}",
            std::any::type_name::<T>()
        ),
        V::Undef | V::Tag(_) => panic!(
            "peek: value is a variant register, not a {}",
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
        V::Erased(any) => match unsafe { Box::from_raw(any) }.downcast::<T>() {
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
        V::None | V::Some(_) => panic!(
            "materialize: value is an option, not a {}",
            std::any::type_name::<T>()
        ),
        V::Undef | V::Tag(_) => panic!(
            "materialize: value is a variant register, not a {}",
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
            V::Closure(c) => (unsafe { &*c.0 })(args),
            V::None
            | V::Undef
            | V::Tag(_)
            | V::Some(_)
            | V::Taken
            | V::Erased(_)
            | V::Reference(_) => {
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
        V::Erased(any) => unsafe { &**any }
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
        V::Erased(any) => unsafe { &mut **any }
            .downcast_mut::<T>()
            .unwrap_or_else(|| panic!("open_mut: value is not a {}", std::any::type_name::<T>())),
        other => panic!("open_mut: not a value: {other:?}"),
    }
}

static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

acvus_extern::cross_one_value!(V, at Tiny);

impl acvus_extern::OneValue<Tiny> for V {
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
    type Op = acvus_extern::DirectOp<Tiny>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = acvus_extern::DirectOp<Tiny>;
    type FusedShape = ();

    acvus_extern::direct_call_forms!();

    fn type_of(&self, value: &V) -> Option<std::any::TypeId> {
        let V::Erased(any) = value else {
            return None;
        };
        Some(unsafe { &**any }.type_id())
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

    fn variant_tag(&self, name: &str) -> V {
        V::Tag(SYMBOLS.intern(name))
    }

    unsafe fn tag_symbol(&self, tag: &V) -> acvus_extern::Astr {
        let V::Tag(name) = tag else {
            panic!("not a tag register: {tag:?}")
        };
        *name
    }

    fn undef(&self) -> V {
        V::Undef
    }

    fn is_undef(&self, value: &V) -> bool {
        matches!(value, V::Undef)
    }

    fn slice_into_run(&self, words: acvus_extern::Words, out: &mut [V]) {
        // A slice is two of the machine's registers (RFC-0047 amended); this
        // runtime has no registers, so each word is its own value.
        out[0] = erased(words.ptr);
        out[1] = erased(words.len);
    }

    unsafe fn slice_from_run(&self, run: &[V]) -> acvus_extern::Words {
        acvus_extern::Words {
            ptr: *open_ref::<u64>(&run[0]),
            len: *open_ref::<u64>(&run[1]),
        }
    }

    type Value = V;
    type Frame<'a> = ();
    type Rooted = ();
    type CallFuture<'a> = Ready<V>;

    fn rooted(&self) {}
    fn frame_of(_: &mut ()) {}

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
    fn none(&self) -> V {
        V::None
    }
    fn some(&self, payload: V) -> V {
        V::Some(Box::into_raw(Box::new(payload)))
    }
    fn is_none(&self, value: &V) -> bool {
        matches!(value, V::None)
    }
    fn unwrap_some(&self, value: V) -> V {
        let V::Some(payload) = value else {
            panic!("unwrap_some: the value is not a Some")
        };
        *unsafe { Box::from_raw(payload) }
    }
    unsafe fn some_at<'a>(&self, value: &'a V) -> Option<&'a V> {
        let V::Some(payload) = value else {
            return None;
        };
        // SAFETY: `some` leaked this payload, and it lives as long as the
        // option that names it.
        Some(unsafe { &**payload })
    }
    unsafe fn some_at_mut<'a>(&self, value: &'a mut V) -> Option<&'a mut V> {
        let V::Some(payload) = value else {
            return None;
        };
        // SAFETY: as `some_at`, with the caller's exclusive loan.
        Some(unsafe { &mut **payload })
    }
    fn call_is_sync(&self, _: &V) -> bool {
        true
    }
    fn call_now<A>(&self, f: &V, _: &mut (), args: A, _: CallToken) -> V
    where
        A: acvus_extern::IntoRun<Self>,
    {
        self.call(f, run_of(self, args))
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
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime;

/// A value that is a source of its own: it carries an identity, so it moves.
#[derive(ExternType)]
#[repr(transparent)]
struct Token<I>(i64, PhantomData<I>)
where
    I: acvus_extern::Var<acvus_extern::kind::Identity>;

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
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
{
    v
}

#[extern_fn(effect = pure)]
async fn apply<T, U, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    v: Boxed<T, E, Rt>,
    f: Fn1<T, U, E, Rt>,
) -> Boxed<U, E, Rt>
where
    T: Var<kind::Type>,
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let f = f.erased();
    let mut out = Vec::with_capacity(v.0.len());
    for item in v.0 {
        out.push(f.call(rt, frame, (item,)).await);
    }
    Boxed(out, PhantomData)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn boxed<T, N, Rt>(rt: &Rt, items: Arr<T, N>) -> Boxed<T, Pure, Rt>
where
    T: Var<kind::Type> + OneValue<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    Boxed(
        items.0.into_iter().map(|v| v.erase(rt)).collect(),
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
    I: acvus_extern::Var<acvus_extern::kind::Identity>,
{
    t.0
}

/// A fresh draw: two draws in either order are the same program.
#[extern_fn(effect = idempotent, commutative)]
fn draw() -> i64 {
    4
}

#[extern_fn(effect = pure, heavy)]
fn digest(s: String) -> i64 {
    s.len() as i64
}

/// Adds `by` to the lent place and returns the new value.
#[extern_fn(effect = pure)]
fn bump(n: &mut i64, by: i64) -> i64 {
    *n += by;
    *n
}

#[extern_fn(effect = pure)]
fn as_slice<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>) -> Slice<T, Rt>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    Slice::of(c.elements(rt))
}

/// The parameter direction of RFC-0047 rule 6: the declaration reads the
/// elements through the view the caller lent, and the pair it was handed is
/// the only thing it was handed.
#[extern_fn(effect = pure)]
fn sum_slice<Rt>(rt: &Rt, s: Slice<i64, Rt>) -> i64
where
    Rt: Runtime,
{
    let elements = s.into_elements();
    (0..elements.len())
        .map(|at| {
            // SAFETY: `at` is below the length the view reports, the
            // container the caller lent is live for the call (RFC-0018), and
            // every element of a language `Vec<i64>` was erased from `i64`.
            unsafe { *rt.value_as_ref::<i64>(elements.at(at)) }
        })
        .sum()
}

extern_signature! { ns: "t", fn eq<T>(a: &T, b: &T) -> bool where T: Var<kind::Type>; }

#[extern_fn(instance_of = eq, effect = pure)]
fn eq_int(a: &i64, b: &i64) -> bool {
    a == b
}

#[extern_fn(instance_of = eq, effect = pure)]
fn eq_point(a: &Point, b: &Point) -> bool {
    a == b
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
        fns: [add, identity, apply, boxed, fetch, digest, take_token, draw, bump, as_slice,
              sum_slice, eq_int, eq_point,
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
        EffectTerm::Known(Effect::OPAQUE.at_task(Task::Async))
    );
    assert_eq!(
        fn_ty(find(&reg.functions, &i, "digest")).effect,
        EffectTerm::Known(Effect::PURE.at_task(Task::Heavy))
    );
    // A declaration that is not pure is split into a `Spawn` and an
    // `Eval` before it runs, and an `Eval` awaits: its task is `Async`
    // however synchronous its Rust body is (RFC-0046).
    assert_eq!(
        fn_ty(find(&reg.functions, &i, "take_token")).effect,
        EffectTerm::Known(Effect::IDEMPOTENT.at_task(Task::Async))
    );
    assert_eq!(
        fn_ty(find(&reg.functions, &i, "draw")).effect,
        EffectTerm::Known(Effect::IDEMPOTENT.commutative().at_task(Task::Async))
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

fn call_sync(handler: &ExternHandler<Tiny>, args: Vec<V>) -> V {
    match handler {
        ExternHandler::Sync(f) => {
            let site = acvus_extern::SitesNoParameterReads::default();
            let f = f.clone().at_site(&site.args(f.arity()));
            // SAFETY: the caller passes the declaration's own arguments.
            unsafe { f.into_op(()).call_run(&Tiny, &args) }
        }
        ExternHandler::Heavy(_) => panic!("expected a sync handler, found a heavy one"),
        ExternHandler::Async(_) => panic!("expected a sync handler, found an async one"),
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

/// The entry hands its run back as the two words the machine keeps a slice
/// in, and those words name the container's own elements in place
/// (RFC-0047 amended, rule 2).
#[test]
fn a_slice_entry_hands_back_two_words_naming_the_container() {
    let (i, reg) = combined::<Tiny>();
    let ExternHandler::Sync(entry) = handler(&reg, &i, "as_slice") else {
        panic!("a declaration returning a slice has a synchronous handler (RFC-0047 §3)")
    };
    assert_eq!(
        entry.width(),
        acvus_extern::Width {
            args: 1,
            ret: 2,
            result: acvus_extern::FormKind::View,
        },
        "a slice-returning declaration takes one container and hands back two words"
    );
    let storage = erased(vec![
        Owned::<Tiny>::from_value(erased(1i64)),
        Owned::<Tiny>::from_value(erased(2i64)),
        Owned::<Tiny>::from_value(erased(3i64)),
    ]);
    // SAFETY: `storage` outlives every run taken from it here (RFC-0018).
    let container = || unsafe { Tiny.reference(&storage) };

    // SAFETY: the handler's width says one argument in and the two words a
    // slice is out.
    let mut pair = [V::default(); 2];
    unsafe {
        entry
            .clone()
            .at_site(&acvus_extern::SitesNoParameterReads::default().args(entry.arity()))
            .into_op(())
            .call(&Tiny, &[container()], &mut pair)
    };
    let Words { ptr, len } = unsafe { Tiny.slice_from_run(&pair) };
    assert_eq!(len, 3);
    assert_ne!(ptr, 0);

    // SAFETY: the words came from the handler, whose storage is live here.
    let run = unsafe { Elements::<Tiny>::from_words(Words { ptr, len }) };
    // SAFETY: as above, over the same two words.
    let rebuilt = unsafe { Elements::<Tiny>::from_words(Words { ptr, len }) };
    for at in 0..run.len() {
        // SAFETY: `at` is below the length the handler reported.
        let (lent, again) = unsafe { (run.at(at), rebuilt.at(at)) };
        assert!(
            std::ptr::eq(lent, again),
            "element {at} is one place in the container's own storage"
        );
        assert!(std::ptr::eq(
            lent,
            &*open_ref::<Vec<Owned<Tiny>>>(&storage)[at]
        ));
    }
}

#[test]
fn a_slice_parameter_is_two_of_the_argument_run_and_reads_the_container() {
    let (i, reg) = combined::<Tiny>();
    let ExternHandler::Sync(entry) = handler(&reg, &i, "sum_slice") else {
        panic!("a declaration taking a slice runs in the caller's frame (RFC-0047 rule 6)")
    };
    assert_eq!(
        entry.width(),
        acvus_extern::Width {
            args: 2,
            ret: 1,
            result: acvus_extern::FormKind::Value,
        },
        "a slice parameter is two of the argument run and the result is one value"
    );
    let storage = vec![erased(4i64), erased(5i64), erased(6i64)];

    let mut run = [V::default(); 2];
    Tiny.slice_into_run(
        Slice::<i64, Tiny>::of(&storage).into_elements().words(),
        &mut run,
    );
    let mut out = [V::default(); 1];
    // SAFETY: `run` is the pair `slice_into_run` just wrote, `storage` is
    // live and unmoved, and `out` has room for the one value the width names.
    unsafe {
        entry
            .clone()
            .at_site(&acvus_extern::SitesNoParameterReads::default().args(entry.arity()))
            .into_op(())
            .call(&Tiny, &run, &mut out)
    };

    assert_eq!(peek::<i64>(&out[0]), 15);
}

async fn call_async(handler: &ExternHandler<Tiny>, args: Vec<V>) -> V {
    match handler {
        ExternHandler::Async(f) => {
            let site = acvus_extern::SitesNoParameterReads::default();
            let f = f.clone().at_site(&site.args(f.arity()));
            // SAFETY: as `call_sync`'s; the future owns `args`.
            unsafe { f.into_op(()).call_async(Tiny, &args) }.await
        }
        ExternHandler::Sync(_) => panic!("expected an async handler, found a sync one"),
        ExternHandler::Heavy(_) => panic!("expected an async handler, found a heavy one"),
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
        .position(|sig| acvus_mir::ty::matches_poly(callee_ty, &sig.ty));
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

    let arr = erased(Arr::<Owned<Tiny>, ()>::new(vec![
        Owned::from_value(erased(1i64)),
        Owned::from_value(erased(2i64)),
    ]));
    let boxed = call_sync(handler(&reg, &i, "boxed"), vec![arr]);
    // SAFETY: `boxed`'s glue erased its return from a `Boxed<T, Pure, Rt>`.
    let Boxed::<Owned<Tiny>, Pure, Tiny>(items, _) =
        unsafe { OneValue::<Tiny>::materialize(&Tiny, boxed) };
    assert_eq!(items.len(), 2);
    let boxed = OneValue::<Tiny>::erase(Boxed::<Owned<Tiny>, (), Tiny>(items, PhantomData), &Tiny);

    let double = V::Closure(Closure::new(|args| {
        let [n] = <[V; 1]>::try_from(args)
            .unwrap_or_else(|args| panic!("double takes one argument, got {}", args.len()));
        erased(open::<i64>(n) * 2)
    }));
    let out = call_async(handler(&reg, &i, "apply"), vec![boxed, double]).await;
    // SAFETY: `apply`'s glue erased its return from a `Boxed<U, E, Rt>`.
    let Boxed::<Owned<Tiny>, (), Tiny>(items, _) =
        unsafe { OneValue::<Tiny>::materialize(&Tiny, out) };
    let doubled: Vec<i64> = items.into_iter().map(open::<i64>).collect();
    assert_eq!(doubled, vec![2, 4]);

    // An object crosses as its fields (RFC-0032): the handler receives
    // `Obj<Owned<Tiny>>` and returns one.
    // The fields are in rule 8's order: `label` before `x`, whatever order the
    // struct declares them in.
    let point = erased(acvus_extern::Obj::new(
        acvus_extern::ObjectShape::of(&SYMBOLS, [Tiny.symbol("x"), Tiny.symbol("label")]),
        Box::new([
            Owned::<Tiny>::from_value(erased("p".to_owned())),
            Owned::from_value(erased(21i64)),
        ]),
    ));
    let out = call_async(handler(&reg, &i, "fetch"), vec![point]).await;
    let obj = open::<acvus_extern::Obj<Owned<Tiny>>>(out);
    let names: Vec<&str> = obj
        .shape
        .names()
        .iter()
        .map(|name| SYMBOLS.resolve(*name))
        .collect();
    assert_eq!(names, vec!["label", "x"]);
    let [label, x] = *<Box<[Owned<Tiny>]> as TryInto<Box<[Owned<Tiny>; 2]>>>::try_into(obj.values)
        .expect("two fields");
    assert_eq!(open::<i64>(x.into_value()), 42);
    assert_eq!(open::<String>(label.into_value()), "p");
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
    let point = acvus_extern::Ty::Object(acvus_extern::ObjectTy::declared(
        i.intern("Point"),
        [
            (i.intern("x"), acvus_extern::Ty::I64),
            (i.intern("label"), acvus_extern::Ty::String),
        ]
        .into_iter()
        .collect(),
    ));
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
    let vars = acvus_extern::PolyVars::fresh(2, 1, 1, 0);
    assert_eq!(
        <Nth<kind::Type, 1> as TyArg>::poly_ty(&i, &vars),
        PolyTy::Var(1)
    );
    assert_eq!(
        <Nth<kind::Effect, 0> as acvus_extern::Term<acvus_extern::kind::Effect>>::poly(&vars),
        EffectTerm::Var(0)
    );
    assert_eq!(
        <Arr<Nth<kind::Type, 0>, Nth<kind::Length, 0>> as TyArg>::poly_ty(&i, &vars),
        PolyTy::Array(Box::new(PolyTy::Var(0)), LenTerm::Var(0))
    );
}

#[derive(TyArg, Debug, PartialEq)]
enum ObjectShape {
    Dot,
    Circle(i64),
    Rect { w: i64, h: i64 },
}

#[test]
fn a_derived_enum_is_the_language_s_enum_of_the_same_name() {
    let i = Interner::new();
    let vars = acvus_extern::PolyVars::fresh(0, 0, 0, 0);
    let rect = PolyTy::Object(acvus_extern::ObjectTy::written(
        [(i.intern("w"), PolyTy::I64), (i.intern("h"), PolyTy::I64)]
            .into_iter()
            .collect(),
    ));
    assert_eq!(
        <ObjectShape as TyArg>::poly_ty(&i, &vars),
        PolyTy::Enum {
            name: i.intern("ObjectShape"),
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
    let vars = acvus_extern::PolyVars::fresh(0, 0, 0, 0);
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

/// A derived extension type read through a reference at a monomorphized
/// member. The parameter names the member, so the crossing is the
/// specialized one, and `&` at it needs the marker the `#[extern_type]`
/// derive states beside the `deref` it writes.
#[extern_fn(effect = pure)]
fn box_width<A, Rt>(v: &Boxed<A, Pure, Rt>) -> i64
where
    A: acvus_extern::Monomorphize<(i64, String)>,
    Rt: Runtime,
{
    v.0.len() as i64
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

#[extern_fn(effect = pure)]
fn twice_ok<A>(r: Result<A, String>) -> Result<A, String>
where
    A: acvus_extern::Monomorphize<(i64, String)> + Twice,
{
    r.map(Twice::twice)
}

fn mono_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        fns: [double, first_or, box_count, box_width, twice_ok],
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

#[test]
fn a_result_of_the_member_type_crosses_a_member_by_value() {
    let i = Interner::new();
    let reg = Externs::combine(vec![mono_registry::<Tiny>()], &i).expect("registries combine");
    let result_of = |t: acvus_extern::Ty| {
        acvus_extern::Ty::Result(Box::new(t), Box::new(acvus_extern::Ty::String))
    };

    let on_int = call_type(
        vec![result_of(acvus_extern::Ty::I64)],
        result_of(acvus_extern::Ty::I64),
        &i,
    );
    let h = instance_for(&reg, &i, "twice_ok", &on_int).expect("the i64 instance");
    assert_eq!(crossed(call_sync(h, vec![crossing(Ok(21i64))])), Ok(42i64));
    assert_eq!(
        crossed::<i64>(call_sync(
            h,
            vec![crossing(Err::<i64, _>(String::from("bad")))]
        )),
        Err(String::from("bad"))
    );

    let on_str = call_type(
        vec![result_of(acvus_extern::Ty::String)],
        result_of(acvus_extern::Ty::String),
        &i,
    );
    let h = instance_for(&reg, &i, "twice_ok", &on_str).expect("the String instance");
    assert_eq!(
        crossed(call_sync(h, vec![crossing(Ok(String::from("ab")))])),
        Ok(String::from("abab"))
    );

    let on_float = call_type(
        vec![result_of(acvus_extern::Ty::Float)],
        result_of(acvus_extern::Ty::Float),
        &i,
    );
    assert!(instance_for(&reg, &i, "twice_ok", &on_float).is_none());
}

/// A `Result` of a member type on its way into a member's argument slot.
fn crossing<A>(r: Result<A, String>) -> V
where
    A: acvus_extern::OneValue<Tiny, acvus_extern::Specialized>,
{
    <Result<A, String> as acvus_extern::OneValue<Tiny, acvus_extern::Specialized>>::erase(r, &Tiny)
}

/// The `Result` a member wrote into its result slot.
fn crossed<A>(value: V) -> Result<A, String>
where
    A: acvus_extern::OneValue<Tiny, acvus_extern::Specialized>,
{
    // SAFETY: the value is what the member's glue wrote through the same
    // crossing.
    unsafe {
        <Result<A, String> as acvus_extern::OneValue<Tiny, acvus_extern::Specialized>>::materialize(
            &Tiny, value,
        )
    }
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
        open::<String>(call_sync(h, vec![V::None, erased(String::from("x"))])),
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
    let payload = OneValue::<Tiny>::erase(
        Boxed::<String, Pure, Tiny>(
            vec![erased(String::from("a")), erased(String::from("b"))],
            PhantomData,
        ),
        &Tiny,
    );
    assert_eq!(open::<i64>(call_sync(h, vec![payload])), 2);
    let ty = call_type(
        vec![boxed_of(acvus_extern::Ty::Float)],
        acvus_extern::Ty::I64,
        &i,
    );
    let fallback = instance_for(&reg, &i, "box_count", &ty).unwrap();
    let payload = OneValue::<Tiny>::erase(
        Boxed::<Owned<Tiny>, Pure, Tiny>(
            vec![erased(1.5f64), erased(2.5f64), erased(3.5f64)],
            PhantomData,
        ),
        &Tiny,
    );
    assert_eq!(open::<i64>(call_sync(fallback, vec![payload])), 3);
}

#[test]
fn a_derived_type_is_read_through_a_reference_at_a_monomorphized_member() {
    let i = Interner::new();
    let reg = Externs::combine(vec![mono_registry::<Tiny>()], &i).expect("registries combine");
    let boxed_of = |t: acvus_extern::Ty| acvus_extern::Ty::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Box")),
        type_args: vec![TypeArg::uniform(t)],
        effect_args: vec![EffectTerm::Known(Effect::PURE)],
        identity_args: vec![],
    };
    let ty = call_type(
        vec![acvus_extern::Ty::Ref(
            acvus_extern::Mutability::Shared,
            Box::new(TypeArg::uniform(boxed_of(acvus_extern::Ty::I64))),
        )],
        acvus_extern::Ty::I64,
        &i,
    );
    let h = instance_for(&reg, &i, "box_width", &ty).expect("an instance on i64");
    let place = OneValue::<Tiny>::erase(
        Boxed::<i64, Pure, Tiny>(vec![erased(1i64), erased(2i64), erased(3i64)], PhantomData),
        &Tiny,
    );
    // SAFETY: `place` outlives the call.
    let r = call_sync(h, vec![unsafe { Tiny.reference(&place) }]);
    assert_eq!(open::<i64>(r), 3);
}

// -- Polymorphic instances (RFC-0027) ---------------------------------

extern_signature! { ns: "t", fn first<C, T>(c: C) -> T where C: Var<kind::Type>, T: Var<kind::Type>; }

#[extern_fn(instance_of = first, effect = pure)]
fn first_arr<T, N>(a: Arr<T, N>) -> T
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
{
    a.0.into_iter().next().expect("first: empty array")
}

#[extern_fn(instance_of = first, effect = pure)]
fn first_opt<T>(v: Option<T>) -> T
where
    T: Var<kind::Type>,
{
    v.expect("first: none")
}

#[extern_fn(instance_of = first, effect = pure)]
fn first_arr_again<T, N>(a: Arr<T, N>) -> T
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
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
    let arr = erased(Arr::<Owned<Tiny>, ()>::new(vec![
        Owned::from_value(erased(7i64)),
        Owned::from_value(erased(8i64)),
    ]));
    let h = instance_for(&reg, &i, "first", &on_array).unwrap();
    assert_eq!(open::<i64>(call_sync(h, vec![arr])), 7);

    let on_option = call_type(
        vec![acvus_extern::Ty::Option(Box::new(acvus_extern::Ty::String))],
        acvus_extern::Ty::String,
        &i,
    );
    let h = instance_for(&reg, &i, "first", &on_option).unwrap();
    assert_eq!(
        open::<String>(call_sync(
            h,
            vec![V::Some(Box::into_raw(Box::new(erased(String::from("s")))))]
        )),
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

// -- A signature declares its effect (RFC-0065) ------------------------

extern_signature! {
    ns: "t",
    effect = E,
    fn drain<C, E>(c: C) -> i64
    where
        C: Var<kind::Type>,
        E: Var<kind::Effect>;
}

extern_signature! { ns: "t", fn size<C>(c: C) -> i64 where C: Var<kind::Type>; }

fn drain_arr_now<T, N, E, Rt>(_: &Rt, _: &mut Rt::Frame<'_>, a: Arr<T, N>) -> i64
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(a.0.len()).expect("an array shorter than i64::MAX")
}

#[extern_fn(instance_of = drain, effect = E, sync = drain_arr_now)]
async fn drain_arr<T, N, E, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, a: Arr<T, N>) -> i64
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    drain_arr_now::<T, N, E, Rt>(rt, frame, a)
}

#[extern_fn(instance_of = drain, effect = pure)]
fn drain_opt<T>(v: Option<T>) -> i64
where
    T: Var<kind::Type>,
{
    i64::from(v.is_some())
}

#[extern_fn(instance_of = size, effect = E)]
fn size_opt<T, E>(v: Option<T>) -> i64
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
{
    i64::from(v.is_some())
}

fn drain_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        signatures: [drain],
        fns: [drain_arr, drain_opt],
    }
}

fn size_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        signatures: [size],
        fns: [size_opt],
    }
}

fn effect_of(reg: &Externs<Tiny>, i: &Interner, name: &str) -> EffectTerm<acvus_extern::Poly> {
    let function = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(i, name))
        .expect("declared");
    let PolyTy::Fn { effect, .. } = &function.ty else {
        panic!("{name} is a function")
    };
    effect.clone()
}

#[test]
fn a_signature_that_declares_an_effect_variable_carries_it() {
    let i = Interner::new();
    let reg = Externs::combine(vec![drain_registry::<Tiny>()], &i).expect("registries combine");
    assert!(
        matches!(effect_of(&reg, &i, "drain"), EffectTerm::Var(_)),
        "{:?}",
        effect_of(&reg, &i, "drain")
    );
    let (i2, pure) = combined::<Tiny>();
    assert_eq!(
        effect_of(&pure, &i2, "eq"),
        EffectTerm::Known(Effect::PURE),
        "a signature that declares no effect is pure"
    );
}

#[test]
fn an_effect_variable_signature_admits_an_effect_instance_and_a_pure_one() {
    let i = Interner::new();
    let reg = Externs::combine(vec![drain_registry::<Tiny>()], &i).expect("registries combine");
    let handlers = &reg.handlers[&qref(&i, "drain")];
    assert_eq!(
        handlers.len(),
        3,
        "the array instance is a Sync/Async pair and the option instance is pure"
    );
    assert_eq!(
        handlers.iter().filter(|h| h.is_sync()).count(),
        2,
        "the pair's plain `fn` and the pure instance reach their result without suspending"
    );

    let on_array = call_type(
        vec![acvus_extern::Ty::Array(
            Box::new(acvus_extern::Ty::I64),
            acvus_extern::LenTerm::Known(2),
        )],
        acvus_extern::Ty::I64,
        &i,
    );
    let arr = erased(Arr::<Owned<Tiny>, ()>::new(vec![
        Owned::from_value(erased(7i64)),
        Owned::from_value(erased(8i64)),
    ]));
    let h = instance_for(&reg, &i, "drain", &on_array).expect("the array instance");
    assert_eq!(open::<i64>(call_sync(h, vec![arr])), 2);

    let on_option = call_type(
        vec![acvus_extern::Ty::Option(Box::new(acvus_extern::Ty::I64))],
        acvus_extern::Ty::I64,
        &i,
    );
    let h = instance_for(&reg, &i, "drain", &on_option).expect("the option instance");
    assert_eq!(
        open::<i64>(call_sync(
            h,
            vec![V::Some(Box::into_raw(Box::new(erased(1i64))))]
        )),
        1
    );
}

#[test]
fn a_pure_signature_refuses_an_effect_variable_instance() {
    let i = Interner::new();
    let err = Externs::combine(vec![size_registry::<Tiny>()], &i)
        .err()
        .expect("an instance generic in its effect would widen a pure signature");
    assert!(
        matches!(err, acvus_extern::CombineError::InstanceMismatch { .. }),
        "{err:?}"
    );
}

// -- A signature names its argument (RFC-0065) -------------------------

extern_signature! {
    ns: "t",
    fn width<T, E, Rt>(b: Boxed<T, E, Rt>) -> i64
    where
        T: Var<kind::Type>,
        E: Var<kind::Effect>,
        Rt: Runtime;
}

#[extern_fn(instance_of = width, effect = pure)]
fn width_i64<E, Rt>(b: Boxed<i64, E, Rt>) -> i64
where
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(b.0.len()).expect("a box shorter than i64::MAX")
}

#[extern_fn(instance_of = width, effect = pure)]
fn width_string<E, Rt>(b: Boxed<String, E, Rt>) -> i64
where
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(b.0.len()).expect("a box shorter than i64::MAX")
}

extern_signature! {
    ns: "t",
    fn head<T, U, E, Rt>(b: Boxed<(T, U), E, Rt>) -> i64
    where
        T: Var<kind::Type>,
        U: Var<kind::Type>,
        E: Var<kind::Effect>,
        Rt: Runtime;
}

#[extern_fn(instance_of = head, effect = pure)]
fn head_i64<U, E, Rt>(b: Boxed<(i64, U), E, Rt>) -> i64
where
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(b.0.len()).expect("a box shorter than i64::MAX")
}

#[extern_fn(instance_of = head, effect = pure)]
fn head_string<U, E, Rt>(b: Boxed<(String, U), E, Rt>) -> i64
where
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(b.0.len()).expect("a box shorter than i64::MAX")
}

extern_signature! { ns: "t", fn present<T>(v: Option<T>) -> i64 where T: Var<kind::Type>; }

#[extern_fn(instance_of = present, effect = pure)]
fn present_i64(v: Option<i64>) -> i64 {
    i64::from(v.is_some())
}

fn named_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        types: [Boxed<_, _, R>],
        signatures: [width, head],
        fns: [width_i64, width_string, head_i64, head_string],
    }
}

fn container_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        signatures: [present],
        fns: [present_i64],
    }
}

fn bound_of(reg: &Externs<Tiny>, i: &Interner, name: &str) -> acvus_extern::TyVarBound {
    let function = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(i, name))
        .expect("declared");
    let acvus_extern::FnKind::Extern { bounds, .. } = &function.kind else {
        panic!("{name} is extern")
    };
    bounds[0].clone()
}

#[test]
fn a_signature_naming_its_argument_is_bounded_by_the_types_inside_it() {
    let i = Interner::new();
    let reg = Externs::combine(vec![named_registry::<Tiny>()], &i).expect("registries combine");
    assert_eq!(
        bound_of(&reg, &i, "width"),
        acvus_extern::TyVarBound::OneOf(vec![PolyTy::I64, PolyTy::String])
    );
    assert_eq!(
        bound_of(&reg, &i, "head"),
        acvus_extern::TyVarBound::OneOf(vec![PolyTy::I64, PolyTy::String])
    );
    assert_eq!(reg.handlers[&qref(&i, "width")].len(), 2);
    assert_eq!(reg.handlers[&qref(&i, "head")].len(), 2);
}

#[test]
fn a_signature_reaching_its_variable_through_a_container_is_refused() {
    let i = Interner::new();
    let err = Externs::combine(vec![container_registry::<Tiny>()], &i)
        .err()
        .expect("an Option is not a head the walk descends");
    assert!(
        matches!(err, acvus_extern::CombineError::InstanceMismatch { .. }),
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

#[test]
fn a_host_some_is_never_none_and_opens_back_to_its_payload() {
    let rt = Tiny;
    assert!(rt.is_none(&rt.none()));
    for depth in 1..4 {
        let mut v = rt.none();
        for _ in 0..depth {
            v = rt.some(v);
        }
        assert!(!rt.is_none(&v), "depth {depth}");
        for _ in 0..depth {
            v = rt.unwrap_some(v);
        }
        assert!(rt.is_none(&v), "depth {depth}");
    }
    let word = rt.some(erased(7i64));
    assert!(!rt.is_none(&word));
    assert_eq!(open::<i64>(rt.unwrap_some(word)), 7);
}

#[test]
fn an_option_crosses_as_the_host_shaped_it() {
    let rt = Tiny;
    let erased_option = <Option<i64> as acvus_extern::OneValue<Tiny>>::erase(Some(4), &rt);
    assert!(matches!(erased_option, V::Some(_)));
    assert_eq!(
        unsafe { <Option<i64> as acvus_extern::OneValue<Tiny>>::materialize(&rt, erased_option) },
        Some(4)
    );
    let nested = <Option<Option<i64>> as acvus_extern::OneValue<Tiny>>::erase(Some(None), &rt);
    assert_eq!(
        unsafe { <Option<Option<i64>> as acvus_extern::OneValue<Tiny>>::materialize(&rt, nested) },
        Some(None)
    );
}

// -- A declaration's task is the ceiling of its handler's (RFC-0046) ---

/// One declaration, built by hand rather than by `#[extern_fn]`, whose
/// type claims `Effect::PURE` — `Task::Sync` — over a handler the runtime
/// offloads and awaits.
fn a_heavy_handler_under_a_pure_declaration() -> Registry<Tiny> {
    Registry::new(|i: &Interner| {
        let qref = acvus_extern::QualifiedRef::qualified(i.intern("t"), i.intern("blocking"));
        let heavy = ExternHandler::heavy(acvus_extern::glue0::<Tiny, _, acvus_extern::Val<V>>(
            |_, _| V::Taken,
        ));
        acvus_extern::Contribution {
            manifest: acvus_extern::Manifest {
                types: Vec::new(),
                signatures: Vec::new(),
                fns: vec![acvus_extern::FnDecl {
                    qref,
                    ty: PolyTy::Fn {
                        params: Vec::new(),
                        ret: Box::new(PolyTy::I64),
                        captures: Vec::new(),
                        effect: EffectTerm::Known(Effect::PURE),
                    },
                    bounds: Vec::new(),
                    cast: false,
                    instance_of: None,
                }],
            },
            instances: acvus_extern::FxHashMap::from_iter([(
                qref,
                acvus_extern::Instances::generic(heavy),
            )]),
            space: acvus_extern::FxHashMap::default(),
        }
    })
}

#[test]
fn a_pure_declaration_over_a_heavy_handler_is_refused() {
    let i = Interner::new();
    let err = Externs::combine(vec![a_heavy_handler_under_a_pure_declaration()], &i)
        .err()
        .expect("a Task::Sync declaration cannot resolve to a Heavy handler");
    assert!(
        matches!(
            err,
            acvus_extern::CombineError::HandlerTask {
                declared: Task::Sync,
                handler: Task::Heavy,
                ..
            }
        ),
        "{err}"
    );
    assert!(
        format!("{err}").contains("blocking"),
        "the refusal names the extern: {err}"
    );
}

// -- `Width` is the library's sum, not the macro's count (RFC-0059) -----

/// Every arity the glue builds reports the width its parameter and result
/// types declare, and the register form at that width reaches the same
/// closure the run form does. Each argument is made fresh: a value crossing
/// by materialization is taken, and one taken twice is freed twice.
#[test]
fn a_glue_reports_the_width_its_types_declare_and_calls_the_same_closure() {
    use acvus_extern::FormKind;
    use acvus_extern::{ByRef, ByValue, Handler, SitesNoParameterReads, Val, Width};

    let site = SitesNoParameterReads::default();

    fn answered<H>(handler: H, expected: Width, args: Vec<V>) -> i64
    where
        H: Handler<Tiny>,
    {
        assert_eq!(H::WIDTH, expected);
        // SAFETY: the arguments are the declaration's own, at its width.
        open::<i64>(unsafe { handler.call_run(&Tiny, (), &args) })
    }

    assert_eq!(
        answered(
            acvus_extern::glue0::<Tiny, _, Val<i64>>(|_, _| 0).at(&site.args(0)),
            Width {
                args: 0,
                ret: 1,
                result: FormKind::Value,
            },
            vec![],
        ),
        0
    );
    assert_eq!(
        answered(
            acvus_extern::glue1::<Tiny, _, ByValue<i64>, Val<i64>>(|_, _, a| a).at(&site.args(1)),
            Width {
                args: 1,
                ret: 1,
                result: FormKind::Value,
            },
            vec![erased(1i64)],
        ),
        1
    );
    assert_eq!(
        answered(
            acvus_extern::glue2::<Tiny, _, ByValue<i64>, ByValue<i64>, Val<i64>>(
                |_, _, a, b| a + b
            ).at(&site.args(2)),
            Width {
                args: 2,
                ret: 1,
                result: FormKind::Value,
            },
            vec![erased(1i64), erased(2i64)],
        ),
        3
    );

    let place = erased(10i64);
    // SAFETY: `place` outlives the reference taken to it here (RFC-0018).
    let lent = unsafe { Tiny.reference(&place) };
    assert_eq!(
        answered(
            acvus_extern::glue3::<Tiny, _, ByValue<i64>, ByRef<i64>, ByValue<i64>, Val<i64>>(
                |_, _, a, b, c| a + *b + c
            )
            .at(&site.args(3)),
            Width {
                args: 3,
                ret: 1,
                result: FormKind::Value,
            },
            vec![erased(1i64), lent, erased(3i64)],
        ),
        14,
        "a borrowed parameter reads the place the reference names"
    );

    assert_eq!(
        answered(
            acvus_extern::glue4::<
                Tiny,
                _,
                ByValue<i64>,
                ByValue<i64>,
                ByValue<i64>,
                ByValue<i64>,
                Val<i64>,
            >(|_, _, a, b, c, d| a + b + c + d)
            .at(&site.args(4)),
            Width {
                args: 4,
                ret: 1,
                result: FormKind::Value,
            },
            vec![erased(1i64), erased(2i64), erased(3i64), erased(4i64)],
        ),
        10
    );

    let two_wide =
        acvus_extern::glue2::<Tiny, _, ByValue<i64>, ByValue<i64>, Val<i64>>(|_, _, a, b| {
            a * 10 + b
        })
        .at(&site.args(2));
    // SAFETY: the width says two arguments in and one value out.
    let by_register = unsafe { two_wide.call2(&Tiny, (), erased(1i64), erased(2i64)) };
    assert_eq!(
        open::<i64>(by_register),
        12,
        "the register form reaches the closure the run form does"
    );
}

/// Every call site holds its own box, cloned out of the one the registry
/// built, and the clone is the same handler: the width it reports and the
/// answer it gives are the original's.
#[test]
fn a_glue_clones_into_a_box_that_is_the_same_handler() {
    use acvus_extern::FormKind;
    use acvus_extern::{ByValue, Handler, HandlerFactory, Val, Width};

    let glue = acvus_extern::glue1::<Tiny, _, ByValue<i64>, Val<i64>>(|_, _, a| a * 3);
    let boxed: Box<dyn HandlerFactory<Tiny>> = Box::new(glue.clone());
    let again = boxed.clone();

    assert_eq!(
        boxed.width(),
        Width {
            args: 1,
            ret: 1,
            result: FormKind::Value,
        }
    );
    assert_eq!(
        again.width(),
        Width {
            args: 1,
            ret: 1,
            result: FormKind::Value,
        }
    );
    let site = acvus_extern::SitesNoParameterReads::default();
    let glue = glue.at(&site.args(1));
    // SAFETY: the width says one argument in and one value out, at each of
    // the three names of this one handler.
    let answers = unsafe {
        [
            glue.call1(&Tiny, (), erased(7i64)),
            boxed
                .at_site(&site.args(1))
                .into_op(())
                .call_run(&Tiny, &[erased(7i64)]),
            again
                .at_site(&site.args(1))
                .into_op(())
                .call_run(&Tiny, &[erased(7i64)]),
        ]
    };
    assert_eq!(answers.map(open::<i64>), [21, 21, 21]);
}

/// A `#[state]` parameter is the closure's own capture, typed: it is no
/// argument of the call, so nothing stands between the call and the value
/// the registry supplied.
#[test]
fn a_state_capture_is_no_argument_of_the_call() {
    let (i, reg) = combined::<Tiny>();
    let ExternHandler::Sync(f) = handler(&reg, &i, "greet") else {
        panic!("greet is a synchronous declaration")
    };
    assert_eq!(
        f.width(),
        acvus_extern::Width {
            args: 1,
            ret: 1,
            result: acvus_extern::FormKind::Value,
        }
    );
}

/// The arguments of a closure call, read back as the `Vec` this runtime's own
/// `call` takes.
fn run_of<Rt, A>(rt: &Rt, args: A) -> Vec<Rt::Value>
where
    Rt: acvus_extern::Runtime,
    A: acvus_extern::IntoRun<Rt>,
{
    let mut run = vec![Rt::Value::default(); A::WIDTH];
    args.into_run(rt, &mut run);
    run
}

/// A parameter that needs nothing from the call site says `Site = ()`, so a
/// declaration of plain parameters carries a site table of zero size and the
/// per-site glue is the closure and nothing else (RFC-0050 rule 6).
#[test]
fn a_plain_declarations_site_table_is_zero_sized() {
    use acvus_extern::{ByValue, SitesNoParameterReads, Val};

    let site = SitesNoParameterReads::default();
    let closure = |_: &Tiny, _: (), a: i64, b: i64| a + b;
    let unsited = acvus_extern::glue2::<Tiny, _, ByValue<i64>, ByValue<i64>, Val<i64>>(closure);
    let sited = unsited.at(&site.args(2));

    assert_eq!(size_of_val(&closure), 0);
    assert_eq!(size_of_val(&sited), 0);
}
