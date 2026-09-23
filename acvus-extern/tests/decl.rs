//! RFC-0023 rule 1 at its contract: one Rust signature yields the acvus type and
//! a handler for any runtime. `Tiny` is a runtime written for this test
//! alone, so nothing here depends on an interpreter.

use acvus_extern::Ctx;
use std::any::Any;
use std::future::Ready;
use std::marker::PhantomData;
use std::ops::Deref;

use acvus_extern::{
    ArgRun, Arr, Borrowable, ClosureFn, Effect, EffectArg, EffectTerm, Erased, ExternHandler,
    ExternType, Externs, Handler, Instance, Interner, LenTerm, Nth, One, OneRegister, OneValue,
    Owned, PolyTy, Pure, Ref, Registry, Runtime, Shared, Task, TransparentOver, TyArg, TypeArg,
    TypesOnly, Var, Words, extern_fn, extern_registry, extern_signature, kind,
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
    /// The language's `Option` (RFC-0039 rule 6), held as a host pleases.
    None,
    /// RFC-0050 rule 8's two variant registers, held as a host pleases.
    Undef,
    Tag(acvus_extern::Astr),
    Some(*mut V),
    Erased(*mut (dyn Any + Send + Sync)),
    Closure(Closure),
    Reference(*const V),
    Instance(*const acvus_extern::InstanceEntry<Tiny>),
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
            V::Taken | V::None | V::Undef | V::Tag(_) | V::Reference(_) | V::Instance(_) => {}
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
        V::Instance(_) => panic!(
            "peek: value is an instance, not a {}",
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
/// A handler's one-value result, as an operation over registers takes it.
///
/// # Safety
/// `Handler::call`'s, at `run`.
unsafe fn one_of<H>(handler: &H, ctx: &mut Ctx<'_, Tiny>, run: &[V]) -> V
where
    H: Handler<Tiny, Ret: OneRegister>,
{
    let rt = ctx.rt;
    let mut out = [V::default()];
    // SAFETY: the caller's contract, which carries the width `from_slice`
    // asks of `run`.
    let verdict = unsafe {
        handler.call(
            ctx,
            <H::Args as ArgRun>::from_slice(run),
            <H::Ret as OneRegister>::slot(&mut out),
        )
    };
    let [written] = out;
    <H::Ret as OneRegister>::land(rt, verdict, written)
}

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
        V::Instance(_) => panic!(
            "materialize: value is an instance, not a {}",
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
            | V::Instance(_)
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
}

impl acvus_extern::Borrowable<Tiny> for V {
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

// SAFETY: `V` is this runtime's own value, which no Rust type disagrees with.
unsafe impl acvus_extern::FromValue<Tiny> for V {
    unsafe fn from_value(_: &Tiny, value: V) -> V {
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
        // A slice is two of the machine's registers (RFC-0047 rule 6); this
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

    fn instance_value(entry: &acvus_extern::InstanceEntry<Tiny>) -> V {
        V::Instance(entry)
    }

    unsafe fn instance_entry<'a>(value: &'a V) -> &'a acvus_extern::InstanceEntry<Tiny> {
        let V::Instance(at) = value else {
            panic!("not an instance: {value:?}")
        };
        // SAFETY: the caller's contract: `instance_value` wrote this value
        // from an entry that outlives `'a`.
        unsafe { &**at }
    }

    type Value = V;
    type Frame<'a> = ();
    type Rooted<'a> = acvus_extern::Ctx<'a, Self>;
    type CallFuture<'a> = Ready<V>;

    fn rooted(&self) -> acvus_extern::Ctx<'_, Self> {
        // SAFETY: the frame is `()`, which names no cells.
        unsafe { acvus_extern::Ctx::new(self, ()) }
    }
    unsafe fn ctx_of<'a, 'r>(
        rooted: &'r mut acvus_extern::Ctx<'a, Self>,
    ) -> &'r mut acvus_extern::Ctx<'a, Self>
    where
        'a: 'r,
    {
        rooted
    }

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
    fn sleep(
        &self,
        d: std::time::Duration,
    ) -> impl std::future::Future<Output = ()> + Send + use<> {
        async move { std::thread::sleep(d) }
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
    unsafe fn call_now<A>(&self, f: &V, _: &mut acvus_extern::Ctx<'_, Self>, args: A) -> V
    where
        A: acvus_extern::IntoRun<Self>,
    {
        self.call(f, run_of(self, args))
    }
    unsafe fn call_0<'a>(&'a self, f: &'a V) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, Vec::new()))
    }
    unsafe fn call_1<'a>(&'a self, f: &'a V, a: V) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, vec![a]))
    }
    unsafe fn call_n<'a>(&'a self, f: &'a V, args: &mut [V]) -> Self::CallFuture<'a> {
        std::future::ready(self.call(f, args.iter_mut().map(std::mem::take).collect()))
    }
}

// -- Declarations under test ------------------------------------------

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Box")]
struct Boxed<T, E, Rt>(Vec<T>, PhantomData<(E, Rt)>)
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
    ctx: &mut Ctx<'_, Rt>,
    v: Boxed<T, E, Rt>,
    f: acvus_extern::Closure<'_, (T,), U, E, Rt>,
) -> Boxed<U, E, Rt>
where
    T: Var<kind::Type> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut out = Vec::with_capacity(v.0.len());
    for item in v.0 {
        out.push(f.call(ctx, (item,)).await);
    }
    Boxed(out, PhantomData)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn boxed<T, N, Rt>(ctx: &mut Ctx<'_, Rt>, items: Arr<T, N>) -> Boxed<T, Pure, Rt>
where
    T: Var<kind::Type> + OneValue<Rt>,
    N: Var<kind::Length>,
    Rt: Runtime,
{
    let _ = ctx;
    Boxed(items.0, PhantomData)
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

/// A returned borrow (RFC-0068 rule 4): the slice is Rust's `&[T]` of the
/// parameter, and the crossing writes the pair.
#[extern_fn(effect = pure)]
fn as_slice<T, Rt>(c: &Vec<T>) -> &[T]
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c
}

/// The parameter direction of RFC-0047 rule 6: the declaration takes the
/// language's `&[i64]` as Rust's slice of the `Erased<Rt, i64>`s it holds.
#[extern_fn(effect = pure)]
fn sum_slice<Rt>(s: &[Erased<Rt, i64>]) -> i64
where
    Rt: Runtime,
{
    s.iter().map(Erased::get).sum()
}

/// A lent result out of a lent slice: the element is in the container the
/// slice borrows, and the crossing writes a reference to it.
#[extern_fn(effect = pure)]
fn slice_first<T, Rt>(s: &[T]) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    s.first()
}

/// A returned borrow of one element, and `None` past the end.
#[extern_fn(effect = pure)]
fn at<T, Rt>(c: &Vec<T>, index: u64) -> Option<&T>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c.get(index as usize)
}

/// A closure parameter declared `&T` is passed as Rust's `&T`: the count
/// of elements the predicate takes.
#[extern_fn(effect = E)]
fn count_where<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    c: &Vec<T>,
    keep: acvus_extern::Closure<'_, (Ref<'static, T, Shared, Rt>,), bool, E, Rt>,
) -> u64
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    c.iter().filter(|x| keep.call_now(ctx, (*x,))).count() as u64
}

extern_signature! { ns: "t", fn eq<T>(a: &T, b: &T) -> bool where T: Var<kind::Type>; }

#[extern_fn(instance_of = eq, effect = pure)]
fn eq_int(a: &i64, b: &i64) -> bool {
    a == b
}

#[extern_fn(instance_of = eq, effect = pure)]
fn eq_string(a: &String, b: &String) -> bool {
    a == b
}

extern_signature! { ns: "t", fn step<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

extern_signature! {
    ns: "t",
    fn front<I, T>(it: &mut I) -> Option<T> where I: Var<kind::Type>, T: Var<kind::Type>;
}

/// A holder of `i64`s at the language's own storage, whose `front` yields a
/// borrow of its first element (RFC-0068 rule 6).
#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Held")]
struct Held<Rt>(Vec<Erased<Rt, i64>>)
where
    Rt: Runtime;

#[extern_fn(effect = pure)]
fn held<Rt>(ctx: &mut Ctx<'_, Rt>, items: Vec<i64>) -> Held<Rt>
where
    Rt: Runtime,
{
    let rt = ctx.rt;
    Held(items.into_iter().map(|n| Erased::new(rt, n)).collect())
}

#[extern_fn(instance_of = front, effect = pure)]
fn front_held<Rt>(it: &mut Held<Rt>) -> Option<&Erased<Rt, i64>>
where
    Rt: Runtime,
{
    it.0.first()
}

/// A requirer that reads a borrowed element out of a `front`: what
/// `Instance::call` hands back at `Ref<Erased<Rt, i64>, Shared, Rt>` is a
/// Rust `&Erased<Rt, i64>` for the receiver's lifetime.
#[extern_fn(effect = pure)]
fn first_of<I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: I,
    front_at: Instance<'_, front<I, Ref<'static, Erased<Rt, i64>, Shared, Rt>, Rt>, I, Rt>,
) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut it = it;
    front_at.call(ctx, &mut it, ()).map(|x| x.get()).unwrap_or(-1)
}

#[extern_fn(instance_of = step, effect = pure)]
fn step_int(n: &mut i64) -> i64 {
    *n += 1;
    *n
}

/// A requiring handler over a signature whose receiver is `&mut I`, with
/// the required parameter itself taken by `&mut`: what `required_at`
/// admits since the site resolves an instance from the settled type of a
/// reference parameter's target.
#[extern_fn(effect = pure)]
fn drive<I, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut I, step_at: Instance<step<I, Rt>, I, Rt>) -> i64
where
    I: Var<kind::Type> + Borrowable<Rt> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    step_at.call(ctx, it, ())
}

/// A requirer that holds its receiver by shared borrow alone (RFC-0070 rule 4).
#[extern_fn(effect = pure)]
fn same<T, Rt>(ctx: &mut Ctx<'_, Rt>, a: &T, b: &T, eq_at: Instance<eq<T, Rt>, T, Rt>) -> bool
where
    T: Var<kind::Type> + Borrowable<Rt> + std::ops::Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    eq_at.call(ctx, a, (&**b,))
}

extern_signature! { ns: "t", fn advance<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

/// An instance that takes an `Instance` parameter, which `#[extern_fn]`
/// refused until RFC-0070 rule 1. It requires `t::step` rather than `t::advance`
/// because an instance that requires its own signature reaches only itself
/// here: a requirement stands at a type variable, so a requiring instance
/// is generic, and a signature admits one generic instance.
#[extern_fn(instance_of = advance, effect = pure)]
fn advance_twice<I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut I,
    inner: Instance<'_, step<I, Rt>, I, Rt>,
) -> i64
where
    I: Var<kind::Type> + Borrowable<Rt> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    inner.call(ctx, &mut *it, ());
    inner.call(ctx, it, ())
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
        types: [Boxed<_, _, R>, Token<_>, Held<R>, Vec<_>],
        signatures: [eq, step, front, advance],
        fns: [add, identity, apply, boxed, fetch, digest, take_token, draw, bump, as_slice,
              sum_slice, slice_first, at, count_where, eq_int, eq_string, step_int, drive,
              held, front_held, first_of, same, advance_twice,
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
        effect_args: vec![EffectArg::uniform(EffectTerm::Var(0))],
        identity_args: vec![],
        region_params: 0,
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
            effect_args: vec![EffectArg::specialized(EffectTerm::Known(Effect::PURE))],
            identity_args: vec![],
            region_params: 0,
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
            let f = f
                .clone()
                .at_site(&acvus_extern::CallSite::of_args(&site.args(f.arity())));
            // SAFETY: the caller passes the declaration's own arguments.
            unsafe { f.into_op(()).call_run(&Tiny, &args) }
        }
        ExternHandler::Heavy(_) => panic!("expected a sync handler, found a heavy one"),
        ExternHandler::Async(_) => panic!("expected a sync handler, found an async one"),
    }
}

/// A `Ctx` over `Tiny`, whose frame is `()`.
fn tiny_ctx() -> Ctx<'static, Tiny> {
    // SAFETY: the frame is `()`, which names no cells.
    unsafe { Ctx::new(&Tiny, ()) }
}

/// One resolved instance, reached the way a site reaches it: the site
/// holds the word of the entry `prepare` built for the requirement, and
/// the receiver is passed at the mode the signature declared.
fn call_instance<'r, S, R>(
    receiver: &Receiver<'_>,
    recv: S::Recv<'r>,
    rest: S::Rest<'r>,
    read: impl FnOnce(S::Ret<'r>) -> R,
) -> R
where
    S: acvus_extern::Signature<Tiny>,
    S::Recv<'r>: acvus_extern::Receiver<Tiny>,
{
    type Site<S> = acvus_extern::Required<S, Place, acvus_extern::Now, 0>;
    let requires = [Tiny::instance_value(&receiver.entry)];
    let args = [receiver.at];
    // SAFETY: the word is `instance_value` of `receiver.entry`, which
    // outlives this call and is the entry `entry_of` built for `S`'s one
    // requirement at the receiver's type.
    let site = unsafe { acvus_extern::CallSite::new(&args, &requires) };
    let instance = <Site<S> as acvus_extern::Sited<Tiny>>::site(&site, 0);
    read(instance.call(&mut tiny_ctx(), recv, rest))
}

/// The site a requirement over one receiver is resolved at: the receiver's
/// settled type, and the entry `prepare` would have built for it.
struct Receiver<'a> {
    at: acvus_extern::ArgAt<'a>,
    entry: acvus_extern::InstanceEntry<Tiny>,
}

/// A storage holding one of `Tiny`'s values: what a requirer's receiver
/// variable is filled with here, as `Owned<Rt>` fills it in a real glue.
struct Place(V);

impl Var<kind::Type> for Place {}

// SAFETY: `Place` holds no `Erased`.
unsafe impl acvus_extern::Canonical<kind::Type> for Place {
    type Canon = Self;
}

impl std::ops::Deref for Place {
    type Target = V;

    fn deref(&self) -> &V {
        &self.0
    }
}

impl std::ops::DerefMut for Place {
    fn deref_mut(&mut self) -> &mut V {
        &mut self.0
    }
}

/// The receiver site of a call of `signature`, whose instance is the `nth`
/// of that signature's own.
fn receiver_at<'a>(
    reg: &'a Externs<Tiny>,
    i: &'a Interner,
    signature: &str,
    call: &'a acvus_extern::Ty,
    nth: usize,
) -> Receiver<'a> {
    let acvus_extern::Ty::Fn { params, .. } = call else {
        panic!("a call type")
    };
    Receiver {
        at: acvus_extern::ArgAt {
            interner: i,
            ty: &params[0].ty,
        },
        entry: entry_of(reg, i, signature, nth),
    }
}

/// The entry `prepare` builds for one instance of `signature`: the glue the
/// registry holds for it, and the words of the entries its own declaration
/// requires.
fn entry_of(
    reg: &Externs<Tiny>,
    i: &Interner,
    signature: &str,
    nth: usize,
) -> acvus_extern::InstanceEntry<Tiny> {
    acvus_extern::InstanceEntry {
        run: <acvus_extern::InstanceTable as acvus_extern::InstanceEntries<Tiny>>::glue(
            &reg.instances,
            qref(i, signature),
            acvus_extern::RequiredInstance(nth),
        ),
        requires: Box::new([]),
    }
}

#[test]
fn a_mono_glue_runs_the_instance_the_glue_runs() {
    let (i, reg) = combined::<Tiny>();
    let on_int = call_type(
        vec![
            acvus_extern::Ty::Ref(
                acvus_extern::Mutability::Shared,
                Box::new(TypeArg::uniform(acvus_extern::Ty::I64)),
            );
            2
        ],
        acvus_extern::Ty::Bool,
        &i,
    );
    let h = instance_for(&reg, &i, "eq", &on_int).expect("the i64 instance of t::eq");
    let (left, right) = (erased(7i64), erased(7i64));
    // SAFETY: both places outlive the two calls below.
    let args = || unsafe { vec![Tiny.reference(&left), Tiny.reference(&right)] };
    assert!(open::<bool>(call_sync(h, args())));
    let recv = Place(left);
    let other = erased(7i64);
    let at = receiver_at(&reg, &i, "eq", &on_int, 0);
    assert!(call_instance::<eq<Place, Tiny>, _>(
        &at,
        &recv,
        (&other,),
        |same| same
    ));
}

#[test]
fn a_receiver_is_named_in_ctx_and_written_through() {
    let (i, reg) = combined::<Tiny>();
    let on_int = call_type(
        vec![acvus_extern::Ty::Ref(
            acvus_extern::Mutability::Mut,
            Box::new(TypeArg::uniform(acvus_extern::Ty::I64)),
        )],
        acvus_extern::Ty::I64,
        &i,
    );
    instance_for(&reg, &i, "step", &on_int).expect("the i64 instance of t::step");
    let mut recv = Place(erased(7i64));
    let at = receiver_at(&reg, &i, "step", &on_int, 0);
    assert_eq!(
        call_instance::<step<Place, Tiny>, _>(&at, &mut recv, (), |n| n),
        8
    );
    assert_eq!(
        call_instance::<step<Place, Tiny>, _>(&at, &mut recv, (), |n| n),
        9
    );
    assert_eq!(peek::<i64>(&recv), 9);
}

/// One requiring handler, called at the settled argument types a site
/// would hand it: the site table resolves every `Instance` parameter from
/// `args`, and the run carries nothing for it.
fn call_requiring(reg: &Externs<Tiny>, i: &Interner, site: RequiringSite<'_>, run: Vec<V>) -> V {
    let ExternHandler::Sync(f) = handler(reg, i, site.name) else {
        panic!("`{}` is declared with a plain `fn`", site.name)
    };
    let at: Vec<acvus_extern::ArgAt<'_>> = site
        .args
        .iter()
        .map(|ty| acvus_extern::ArgAt { interner: i, ty })
        .collect();
    // SAFETY: every caller fills `site.requires` with `instance_value` of
    // the `entry_of` each requirement of `site.name` is resolved to, and
    // keeps those entries alive across this call.
    let f = f
        .clone()
        .at_site(&unsafe { acvus_extern::CallSite::new(&at, site.requires) });
    // SAFETY: the run is the declaration's own arguments at the types the
    // site table was filled from.
    unsafe { f.into_op(()).call_run(&Tiny, &run) }
}

/// A call of a requiring declaration as a site states it: the settled
/// argument types, and the checker's answer for each requirement.
struct RequiringSite<'a> {
    name: &'a str,
    args: &'a [acvus_extern::Ty],
    requires: &'a [V],
}

/// `drive` takes the parameter its requirement stands at by `&mut`, so the
/// site's settled type is `&mut i64`: `InstanceTable::instance_at` reads
/// that as the instance at `i64`, and `t::step` bumps the place the caller
/// lent.
#[test]
fn a_required_instance_resolves_from_a_reference_parameters_target() {
    let (i, reg) = combined::<Tiny>();
    let place = erased(7i64);
    // SAFETY: `place` outlives the call below.
    let lent = unsafe { Tiny.reference(&place) };
    let out = call_requiring(
        &reg,
        &i,
        RequiringSite {
            name: "drive",
            args: &[acvus_extern::Ty::Ref(
                acvus_extern::Mutability::Mut,
                Box::new(TypeArg::uniform(acvus_extern::Ty::I64)),
            )],
            requires: &[Tiny::instance_value(&entry_of(&reg, &i, "step", 0))],
        },
        vec![lent],
    );
    assert_eq!(
        open::<i64>(out),
        8,
        "t::step at i64 bumps the place it lends"
    );
    assert_eq!(peek::<i64>(&place), 8, "and the caller sees the write");
}

/// A requirer holding a shared borrow calls the instance it requires
/// (RFC-0070 rule 4).
#[test]
fn a_shared_receiver_reaches_the_instance_it_requires() {
    let (i, reg) = combined::<Tiny>();
    let (left, right) = (erased(7i64), erased(8i64));
    // SAFETY: both places outlive the calls below.
    let (a, b) = unsafe { (Tiny.reference(&left), Tiny.reference(&right)) };
    let shared_i64 = acvus_extern::Ty::Ref(
        acvus_extern::Mutability::Shared,
        Box::new(TypeArg::uniform(acvus_extern::Ty::I64)),
    );
    let eq_entry = entry_of(&reg, &i, "eq", 0);
    let eq_at = [Tiny::instance_value(&eq_entry)];
    let same = |run: Vec<V>| {
        open::<bool>(call_requiring(
            &reg,
            &i,
            RequiringSite {
                name: "same",
                args: &[shared_i64.clone(), shared_i64.clone()],
                requires: &eq_at,
            },
            run,
        ))
    };
    assert!(!same(vec![a, b]));
    assert!(same(vec![a, a]));
}

/// An instance states what it requires on its own candidate, which is
/// where the checker reads it (RFC-0070 rule 1).
#[test]
fn an_instance_carries_its_own_requirements_and_keeps_its_mono_glue() {
    let (i, reg) = combined::<Tiny>();
    let declared = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(&i, "advance"))
        .expect("t::advance is declared");
    let acvus_extern::FnKind::Extern {
        instances,
        requires,
        ..
    } = &declared.kind
    else {
        panic!("an extern declaration")
    };
    assert!(
        requires.is_empty(),
        "a shared signature requires nothing of its own"
    );
    let [advance_twice] = instances.concrete.as_slice() else {
        panic!("t::advance has one instance")
    };
    assert_eq!(
        advance_twice
            .requires
            .iter()
            .map(|r| r.signature)
            .collect::<Vec<_>>(),
        vec![qref(&i, "step")]
    );
    assert!(
        handler(&reg, &i, "advance").instance().is_some(),
        "a requiring instance is reached through a mono glue like any other"
    );

    let mut recv = Place(erased(7i64));
    let step_entry = entry_of(&reg, &i, "step", 0);
    let on_int = call_type(
        vec![acvus_extern::Ty::Ref(
            acvus_extern::Mutability::Mut,
            Box::new(TypeArg::uniform(acvus_extern::Ty::I64)),
        )],
        acvus_extern::Ty::I64,
        &i,
    );
    let at = Receiver {
        at: acvus_extern::ArgAt {
            interner: &i,
            ty: &on_int,
        },
        entry: acvus_extern::InstanceEntry {
            run: <acvus_extern::InstanceTable as acvus_extern::InstanceEntries<Tiny>>::glue(
                &reg.instances,
                qref(&i, "advance"),
                acvus_extern::RequiredInstance(0),
            ),
            requires: Box::new([Tiny::instance_value(&step_entry)]),
        },
    };
    assert_eq!(
        call_instance::<advance<Place, Tiny>, _>(&at, &mut recv, (), |n| n),
        9,
        "the glue bound its requirement from the entry and ran it twice"
    );
}

/// The marker `extern_signature!` writes names the signature's own
/// variables, so a handler writes the requirement at its own: `eq<T, Rt>`.
/// Every parameter is defaulted, so the bare `eq` an `instance_of`
/// attribute names still resolves.
#[test]
fn a_marker_names_its_variables() {
    let i = Interner::new();
    fn qref_of<S>(i: &Interner) -> acvus_extern::QualifiedRef
    where
        S: acvus_extern::SharedSignature,
    {
        S::qref(i)
    }
    assert_eq!(qref_of::<eq>(&i), qref_of::<eq<i64, Tiny>>(&i));
    assert_eq!(i.resolve(qref_of::<eq>(&i).name), "eq");
}

#[test]
fn a_declaration_that_is_no_instance_has_no_mono_glue() {
    let (i, reg) = combined::<Tiny>();
    assert!(handler(&reg, &i, "add").instance().is_none());
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
/// (RFC-0047 rule 6).
#[test]
fn a_slice_entry_hands_back_two_words_naming_the_container() {
    let (i, reg) = combined::<Tiny>();
    let ExternHandler::Sync(entry) = handler(&reg, &i, "as_slice") else {
        panic!("a declaration returning a slice has a synchronous handler (RFC-0023 rule 6)")
    };
    assert_eq!(
        entry.width(),
        acvus_extern::Width {
            args: 1,
            ret: 2,
            result: acvus_extern::FormKind::View,
            absent: false,
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
            .at_site(&acvus_extern::CallSite::of_args(
                &acvus_extern::SitesNoParameterReads::default().args(entry.arity()),
            ))
            .into_op(())
            .call(&Tiny, &[container()], &mut pair)
    };
    let Words { ptr, len } = unsafe { Tiny.slice_from_run(&pair) };
    assert_eq!(len, 3);
    assert_ne!(ptr, 0);

    let run = ptr as *const V;
    for at in 0..len as usize {
        // SAFETY: the words came from the handler, whose storage is live
        // here, and `at` is below the length the handler reported.
        let (lent, again) = unsafe { (&*run.add(at), &*run.add(at)) };
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
            absent: false,
        },
        "a slice parameter is two of the argument run and the result is one value"
    );
    let storage = vec![erased(4i64), erased(5i64), erased(6i64)];

    let mut run = [V::default(); 2];
    Tiny.slice_into_run(
        Words {
            ptr: storage.as_ptr() as u64,
            len: storage.len() as u64,
        },
        &mut run,
    );
    let mut out = [V::default(); 1];
    // SAFETY: `run` is the pair `slice_into_run` just wrote, `storage` is
    // live and unmoved, and `out` has room for the one value the width names.
    unsafe {
        entry
            .clone()
            .at_site(&acvus_extern::CallSite::of_args(
                &acvus_extern::SitesNoParameterReads::default().args(entry.arity()),
            ))
            .into_op(())
            .call(&Tiny, &run, &mut out)
    };

    assert_eq!(peek::<i64>(&out[0]), 15);
}

async fn call_async(handler: &ExternHandler<Tiny>, args: Vec<V>) -> V {
    match handler {
        ExternHandler::Async(f) => {
            let site = acvus_extern::SitesNoParameterReads::default();
            let f = f
                .clone()
                .at_site(&acvus_extern::CallSite::of_args(&site.args(f.arity())));
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
    let index = concrete.or_else(|| instances.generic.as_ref().map(|_| instances.generic_index()))?;
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
    let doubled: Vec<i64> = items
        .into_iter()
        .map(|item| open::<i64>(item.into_value()))
        .collect();
    assert_eq!(doubled, vec![2, 4]);

    // An object crosses as its fields (RFC-0039 rule 4): the handler receives
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

/// A lent result (RFC-0068 rule 4): `at` returns Rust's `Option<&T>` of the
/// container the caller lent, and the crossing writes a reference into
/// that container's own storage, or `None`.
#[test]
fn a_lent_element_names_the_containers_own_storage() {
    let (i, reg) = combined::<Tiny>();
    let storage = erased(vec![
        Owned::<Tiny>::from_value(erased(10i64)),
        Owned::<Tiny>::from_value(erased(20i64)),
    ]);
    // SAFETY: `storage` outlives every reference taken from it here.
    let container = || unsafe { Tiny.reference(&storage) };
    let found = call_sync(handler(&reg, &i, "at"), vec![container(), erased(1u64)]);
    let V::Some(inner) = found else {
        panic!("index 1 of two elements is found: {found:?}")
    };
    // SAFETY: `Tiny::some` boxed the payload and nothing freed it.
    let element = target(unsafe { &*inner });
    assert_eq!(peek::<i64>(element), 20);
    let past = call_sync(handler(&reg, &i, "at"), vec![container(), erased(2u64)]);
    assert!(
        matches!(past, V::None),
        "index 2 of two elements is None: {past:?}"
    );
}

/// A signature whose instance yields a borrow of its receiver: the glue
/// crosses the `&Erased<Rt, i64>` as a reference word, and the requirer
/// reads it back as a Rust borrow of the receiver it lent (RFC-0068 rule 6).
#[test]
fn a_lent_yield_reaches_the_requirer_as_a_borrow_of_its_receiver() {
    let (i, reg) = combined::<Tiny>();
    let held = call_sync(
        handler(&reg, &i, "held"),
        vec![erased(vec![
            Owned::<Tiny>::from_value(erased(41i64)),
            Owned::<Tiny>::from_value(erased(5i64)),
        ])],
    );
    let held_ty = acvus_extern::Ty::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Held")),
        type_args: vec![],
        effect_args: vec![],
        identity_args: vec![],
        region_params: 0,
    };
    let front_entry = entry_of(&reg, &i, "front", 0);
    let front_at = [Tiny::instance_value(&front_entry)];
    fn first_of<'a>(args: &'a [acvus_extern::Ty], front_at: &'a [V]) -> RequiringSite<'a> {
        RequiringSite {
            name: "first_of",
            args,
            requires: front_at,
        }
    }
    let out = call_requiring(
        &reg,
        &i,
        first_of(&[held_ty.clone()], &front_at),
        vec![held],
    );
    assert_eq!(open::<i64>(out), 41);
    let empty = call_sync(
        handler(&reg, &i, "held"),
        vec![erased(Vec::<Owned<Tiny>>::new())],
    );
    let out = call_requiring(&reg, &i, first_of(&[held_ty], &front_at), vec![empty]);
    assert_eq!(open::<i64>(out), -1);
}

/// A slice parameter taken as Rust's `&[T]`, and a lent element out of it:
/// the pair the caller wrote names the container, so the element the
/// handler returns is in that container's own storage.
#[test]
fn a_lent_element_out_of_a_lent_slice_names_the_container() {
    let (i, reg) = combined::<Tiny>();
    let storage = vec![erased(7i64), erased(8i64)];
    let mut pair = [V::default(); 2];
    Tiny.slice_into_run(
        Words {
            ptr: storage.as_ptr() as u64,
            len: storage.len() as u64,
        },
        &mut pair,
    );
    let ExternHandler::Sync(entry) = handler(&reg, &i, "slice_first") else {
        panic!("a declaration returning a borrow has a synchronous handler")
    };
    let mut out = [V::default(); 1];
    // SAFETY: `pair` is the pair `slice_into_run` just wrote, `storage` is
    // live and unmoved, and `out` has room for the one value the width names.
    unsafe {
        entry
            .clone()
            .at_site(&acvus_extern::CallSite::of_args(
                &acvus_extern::SitesNoParameterReads::default().args(entry.arity()),
            ))
            .into_op(())
            .call(&Tiny, &pair, &mut out)
    };
    let [found] = out;
    let V::Some(inner) = found else {
        panic!("the first of two elements is found: {found:?}")
    };
    // SAFETY: `Tiny::some` boxed the payload and nothing freed it.
    let element = target(unsafe { &*inner });
    assert!(
        std::ptr::eq(element, &storage[0]),
        "the lent element is the container's own first slot"
    );
}

/// A closure parameter declared `&T` is passed as Rust's `&T` and reaches
/// the closure as a reference into the container the handler borrowed.
#[test]
fn a_borrowed_closure_argument_is_a_reference_into_the_handlers_borrow() {
    let (i, reg) = combined::<Tiny>();
    let storage = erased(vec![
        Owned::<Tiny>::from_value(erased(1i64)),
        Owned::<Tiny>::from_value(erased(5i64)),
        Owned::<Tiny>::from_value(erased(9i64)),
    ]);
    // SAFETY: `storage` outlives every reference taken from it here.
    let container = unsafe { Tiny.reference(&storage) };
    let above_three = V::Closure(Closure::new(|args| {
        let [x] = <[V; 1]>::try_from(args)
            .unwrap_or_else(|args| panic!("the predicate takes one argument, got {}", args.len()));
        erased(peek::<i64>(target(&x)) > 3)
    }));
    let out = call_sync(
        handler(&reg, &i, "count_where"),
        vec![container, above_three],
    );
    assert_eq!(open::<u64>(out), 2);
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
    let eq_fn = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(&i, "eq"))
        .expect("eq");
    let acvus_extern::FnKind::Extern { bounds, .. } = &eq_fn.kind else {
        panic!("eq is extern")
    };
    let acvus_extern::TyVarBound::OneOf { shapes } = &bounds[0] else {
        panic!("eq's first bound is an instance set")
    };
    assert_eq!(
        shapes,
        &vec![acvus_extern::PolyTy::I64, acvus_extern::PolyTy::String]
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
    assert!(matches!(
        err,
        acvus_extern::CombineError::DuplicateName { .. }
    ));
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

/// A closure's acvus type is read off the parameter tuple, and it is the type
/// the per-arity carriers printed: the slot names are the positions `_0`,
/// `_1`, … in tuple order, and the effect and the return follow them.
#[test]
fn a_closure_type_is_its_parameter_tuple_in_order() {
    let i = Interner::new();
    let vars = acvus_extern::PolyVars::fresh(0, 0, 0, 0);

    assert_eq!(
        <acvus_extern::Closure<(i64, bool), String, Pure, TypesOnly> as TyArg>::poly_ty(&i, &vars),
        PolyTy::Fn {
            params: vec![
                acvus_extern::ParamTerm::<acvus_extern::Poly>::new(
                    i.intern("_0"),
                    <i64 as TyArg>::poly_ty(&i, &vars)
                ),
                acvus_extern::ParamTerm::<acvus_extern::Poly>::new(
                    i.intern("_1"),
                    <bool as TyArg>::poly_ty(&i, &vars)
                ),
            ],
            ret: Box::new(<String as TyArg>::poly_ty(&i, &vars)),
            captures: vec![],
            effect: <Pure as acvus_extern::Term<kind::Effect>>::poly(&vars),
        }
    );

    assert_eq!(
        <acvus_extern::Closure<(), i64, Pure, TypesOnly> as TyArg>::poly_ty(&i, &vars),
        PolyTy::Fn {
            params: vec![],
            ret: Box::new(<i64 as TyArg>::poly_ty(&i, &vars)),
            captures: vec![],
            effect: <Pure as acvus_extern::Term<kind::Effect>>::poly(&vars),
        }
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
            name: acvus_extern::QualifiedRef::root(i.intern("ObjectShape")),
            variants: [
                (i.intern("Dot"), None),
                (i.intern("Circle"), Some(Box::new(PolyTy::I64))),
                (i.intern("Rect"), Some(Box::new(rect))),
            ]
            .into_iter()
            .collect(),
            home: acvus_extern::Home::NONE,
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
        types: [Boxed<_, _, R>],
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
        &vec![acvus_extern::TyVarBound::one_of(vec![
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

    let boxed_of = |arg: TypeArg<acvus_mir::ty::Concrete>| acvus_extern::Ty::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Box")),
        type_args: vec![arg],
        effect_args: vec![EffectArg::specialized(EffectTerm::Known(Effect::PURE))],
        identity_args: vec![],
        region_params: 0,
    };
    let ty = call_type(
        vec![boxed_of(TypeArg::specialized(acvus_extern::Ty::String))],
        acvus_extern::Ty::I64,
        &i,
    );
    let h = instance_for(&reg, &i, "box_count", &ty).unwrap();
    let payload = OneValue::<Tiny>::erase(
        Boxed::<String, Pure, Tiny>(vec![String::from("a"), String::from("b")], PhantomData),
        &Tiny,
    );
    assert_eq!(open::<i64>(call_sync(h, vec![payload])), 2);
    let ty = call_type(
        vec![boxed_of(TypeArg::uniform(acvus_extern::Ty::Float))],
        acvus_extern::Ty::I64,
        &i,
    );
    let fallback = instance_for(&reg, &i, "box_count", &ty).unwrap();
    let payload = OneValue::<Tiny>::erase(
        Boxed::<Owned<Tiny>, Pure, Tiny>(
            vec![
                Owned::from_value(erased(1.5f64)),
                Owned::from_value(erased(2.5f64)),
                Owned::from_value(erased(3.5f64)),
            ],
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
    let boxed_of = |arg: TypeArg<acvus_mir::ty::Concrete>| acvus_extern::Ty::UserDefined {
        id: acvus_extern::QualifiedRef::root(i.intern("Box")),
        type_args: vec![arg],
        effect_args: vec![EffectArg::specialized(EffectTerm::Known(Effect::PURE))],
        identity_args: vec![],
        region_params: 0,
    };
    let ty = call_type(
        vec![acvus_extern::Ty::Ref(
            acvus_extern::Mutability::Shared,
            Box::new(TypeArg::uniform(boxed_of(TypeArg::specialized(
                acvus_extern::Ty::I64,
            )))),
        )],
        acvus_extern::Ty::I64,
        &i,
    );
    let h = instance_for(&reg, &i, "box_width", &ty).expect("an instance on i64");
    let place = OneValue::<Tiny>::erase(
        Boxed::<i64, Pure, Tiny>(vec![1i64, 2i64, 3i64], PhantomData),
        &Tiny,
    );
    // SAFETY: `place` outlives the call.
    let r = call_sync(h, vec![unsafe { Tiny.reference(&place) }]);
    assert_eq!(open::<i64>(r), 3);
}

// -- Polymorphic instances (RFC-0019) ---------------------------------

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
    let acvus_extern::FnKind::Extern {
        bounds, instances, ..
    } = &first_fn.kind
    else {
        panic!("first is extern")
    };
    assert_eq!(instances.concrete.len(), 2);
    assert!(instances.generic.is_none());
    let acvus_extern::TyVarBound::OneOf { shapes, .. } = &bounds[0] else {
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

// -- A signature declares its effect (RFC-0067 rule 2) ------------------------

extern_signature! {
    ns: "t",
    effect = E,
    fn drain<C, E>(c: C) -> i64
    where
        C: Var<kind::Type>,
        E: Var<kind::Effect>;
}

extern_signature! { ns: "t", fn size<C>(c: C) -> i64 where C: Var<kind::Type>; }

fn drain_arr_now<T, N, E, Rt>(_: &mut Ctx<'_, Rt>, a: Arr<T, N>) -> i64
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(a.0.len()).expect("an array shorter than i64::MAX")
}

#[extern_fn(instance_of = drain, effect = E, sync = drain_arr_now)]
async fn drain_arr<T, N, E, Rt>(ctx: &mut Ctx<'_, Rt>, a: Arr<T, N>) -> i64
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    drain_arr_now::<T, N, E, Rt>(ctx, a)
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

// -- A signature names its argument (RFC-0067 rule 2) -------------------------

extern_signature! {
    ns: "t",
    fn width<T, E, Rt>(b: Boxed<T, E, Rt>) -> i64
    where
        T: Var<kind::Type>,
        E: Var<kind::Effect>,
        Rt: Runtime;
}

#[extern_fn(instance_of = width, effect = pure)]
fn width_i64<E, Rt>(b: Boxed<Erased<Rt, i64>, E, Rt>) -> i64
where
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(b.0.len()).expect("a box shorter than i64::MAX")
}

#[extern_fn(instance_of = width, effect = pure)]
fn width_string<E, Rt>(b: Boxed<Erased<Rt, String>, E, Rt>) -> i64
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

/// The signature's `(T, U)` is held part by part, `T` uniform, so an
/// instance holds its `T` as the erased value the signature's box holds
/// there, as `width_i64` does at the top.
#[extern_fn(instance_of = head, effect = pure)]
fn head_i64<U, E, Rt>(b: Boxed<(Erased<Rt, i64>, U), E, Rt>) -> i64
where
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(b.0.len()).expect("a box shorter than i64::MAX")
}

#[extern_fn(instance_of = head, effect = pure)]
fn head_string<U, E, Rt>(b: Boxed<(Erased<Rt, String>, U), E, Rt>) -> i64
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

/// The shapes of a declaration's first bound.
fn bound_of(reg: &Externs<Tiny>, i: &Interner, name: &str) -> Vec<PolyTy> {
    let function = reg
        .functions
        .iter()
        .find(|f| f.qref == qref(i, name))
        .expect("declared");
    let acvus_extern::FnKind::Extern { bounds, .. } = &function.kind else {
        panic!("{name} is extern")
    };
    let acvus_extern::TyVarBound::OneOf { shapes } = &bounds[0] else {
        panic!("{name}'s first bound is an instance set")
    };
    shapes.clone()
}

#[test]
fn a_signature_naming_its_argument_is_bounded_by_the_types_inside_it() {
    let i = Interner::new();
    let reg = Externs::combine(vec![named_registry::<Tiny>()], &i).expect("registries combine");
    assert_eq!(
        bound_of(&reg, &i, "width"),
        vec![PolyTy::I64, PolyTy::String]
    );
    assert_eq!(
        bound_of(&reg, &i, "head"),
        vec![PolyTy::I64, PolyTy::String]
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

// -- A signature declares the variables its instances choose (RFC-0041) -------

extern_signature! {
    ns: "t",
    fn spread<T, E, Rt>(b: Boxed<T, E, Rt>) -> i64
    where
        T: Var<kind::Type>,
        E: Var<kind::Effect>,
        Rt: Runtime;
}

#[extern_fn(instance_of = spread, effect = pure)]
fn spread_pair<U, E, Rt>(b: Boxed<(i64, U), E, Rt>) -> i64
where
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(b.0.len()).expect("a box shorter than i64::MAX")
}

extern_signature! {
    ns: "t",
    fn chosen_spread<T, E, Rt>(b: Boxed<T, E, Rt>) -> i64
    where
        T: Var<kind::Type> + Chosen,
        E: Var<kind::Effect>,
        Rt: Runtime;
}

#[extern_fn(instance_of = chosen_spread, effect = pure)]
fn chosen_spread_pair<U, E, Rt>(b: Boxed<(i64, U), E, Rt>) -> i64
where
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    i64::try_from(b.0.len()).expect("a box shorter than i64::MAX")
}

fn undeclared_slot_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        types: [Boxed<_, _, R>],
        signatures: [spread],
        fns: [spread_pair],
    }
}

fn chosen_slot_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "t",
        types: [Boxed<_, _, R>],
        signatures: [chosen_spread],
        fns: [chosen_spread_pair],
    }
}

#[test]
fn an_undeclared_variable_at_a_held_slot_is_uniform_and_refuses_a_specialized_instance() {
    let i = Interner::new();
    let err = Externs::combine(vec![undeclared_slot_registry::<Tiny>()], &i)
        .err()
        .expect("`Box<#(#i64, U)>` is not the uniform `Box<T>`");
    assert!(
        matches!(err, acvus_extern::CombineError::InstanceMismatch { .. }),
        "{err:?}"
    );
}

#[test]
fn a_chosen_variable_at_a_held_slot_takes_the_instance_s_tree() {
    let i = Interner::new();
    let reg = Externs::combine(vec![chosen_slot_registry::<Tiny>()], &i)
        .expect("the instance chooses `T` as `#(#i64, U)`");
    assert_eq!(reg.handlers[&qref(&i, "chosen_spread")].len(), 1);
}

/// An `Inline` element is read, copied and edited through `Erased` with no
/// runtime in hand; a `String` still needs one.
#[test]
fn erased_inline_derefs_without_a_runtime() {
    use acvus_extern::Erased;
    let rt = Tiny;
    let mut n: Erased<Tiny, i64> = Erased::new(&rt, 41);
    assert_eq!(*n.get_ref(), 41);
    assert_eq!(n.get(), 41);
    *n.get_mut() += 1;
    assert_eq!(n.get(), 42);
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

/// The nesting `Value::some` counts is the operation's to build once the
/// verdict leaves the handler: `Some(None)` is a `some` over the inner
/// option's own `none`, and the outer `None` is that `none` alone.
#[test]
fn an_absent_result_lands_as_the_operation_wraps_it_and_keeps_its_nesting() {
    use acvus_extern::{ByValue, OptionOf, Val};

    let site = acvus_extern::SitesNoParameterReads::default();
    let glue =
        acvus_extern::glue::<Tiny, _, (ByValue<bool>,), Val<Option<Option<i64>>>>(|_, (there,)| {
            match there {
                true => Some(None),
                false => None,
            }
        })
        .at(&acvus_extern::CallSite::of_args(&site.args(1)));

    /// The bound is the assertion that `Option<T>` leaves the handler at
    /// `OptionOf<One>`: a glue of any other result form does not reach here.
    ///
    /// # Safety
    /// `one_of`'s, at one argument.
    unsafe fn landed<H>(handler: &H, arg: V) -> V
    where
        H: Handler<Tiny, Ret = OptionOf<One>>,
    {
        assert_eq!(H::WIDTH.ret, 1);
        assert_eq!(H::WIDTH.result, acvus_extern::FormKind::Value);
        // SAFETY: the caller's contract.
        unsafe { one_of(handler, &mut tiny_ctx(), &[arg]) }
    }

    // SAFETY: the width says one argument in and one value out.
    let some_none = unsafe { landed(&glue, erased(true)) };
    // SAFETY: as above.
    let none = unsafe { landed(&glue, erased(false)) };

    let rt = Tiny;
    assert!(
        rt.is_none(&none),
        "the absent result lands as the host's none"
    );
    assert!(
        !rt.is_none(&some_none),
        "a present result lands as the host's some, whatever it holds"
    );
    assert!(
        rt.is_none(&rt.unwrap_some(some_none)),
        "the inner option's own none survives under the some the operation built"
    );
}

// -- A declaration's task is the ceiling of its handler's (RFC-0046) ---

/// One declaration, built by hand rather than by `#[extern_fn]`, whose
/// type claims `Effect::PURE` — `Task::Sync` — over a handler the runtime
/// offloads and awaits.
fn a_heavy_handler_under_a_pure_declaration() -> Registry<Tiny> {
    Registry::new(|i: &Interner| {
        let qref = acvus_extern::QualifiedRef::qualified(i.intern("t"), i.intern("blocking"));
        let heavy = ExternHandler::heavy(acvus_extern::glue::<Tiny, _, (), acvus_extern::Val<V>>(
            |_, ()| V::Taken,
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
                    effect_bounds: vec![],
                    coercion: None,
                    instance_of: None,
                    requires: Vec::new(),
                    names: Vec::new(),
                    laws: acvus_extern::Laws::None,
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

// -- A requirement reaches a mono glue or it is refused (RFC-0067) -----

/// What a stateful instance holds, and what makes it no plain `fn`.
struct By(i64);

#[extern_fn(instance_of = step, effect = pure)]
fn step_stated(#[state] by: &By, s: &mut String) -> i64 {
    s.push('x');
    by.0 + s.len() as i64
}

fn a_stateful_instance_beside_a_requirement<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "t",
        types: [],
        signatures: [step],
        fns: [step_int, step_stated(By(1)), drive],
    }
}

/// `drive` requires `t::step`, so every instance of `t::step` is reached
/// through a mono glue; `step_stated` holds a `#[state]` value and has
/// none. The fact is first known where the two meet, which is `combine` —
/// `#[extern_fn]` knows the shape but not that anything requires it.
#[test]
fn a_required_signature_whose_instance_has_no_mono_glue_is_refused_at_combine() {
    let i = Interner::new();
    let err = Externs::<Tiny>::combine(vec![a_stateful_instance_beside_a_requirement()], &i)
        .err()
        .expect("t::step is required and one of its instances holds a state");
    assert!(
        matches!(
            err,
            acvus_extern::CombineError::RequiredInstanceWithoutGlue { .. }
        ),
        "{err}"
    );
    let written = format!("{err}");
    assert!(
        written.contains("t::step_stated") && written.contains("`#[state]`"),
        "the refusal names the instance and the shapes that have no glue: {written}"
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
        H: Handler<Tiny, Ret = One>,
    {
        assert_eq!(H::WIDTH, expected);
        // SAFETY: the arguments are the declaration's own, at its width.
        open::<i64>(unsafe { one_of(&handler, &mut tiny_ctx(), &args) })
    }

    assert_eq!(
        answered(
            acvus_extern::glue::<Tiny, _, (), Val<i64>>(|_, ()| 0)
                .at(&acvus_extern::CallSite::of_args(&site.args(0))),
            Width {
                args: 0,
                ret: 1,
                result: FormKind::Value,
                absent: false,
            },
            vec![],
        ),
        0
    );
    assert_eq!(
        answered(
            acvus_extern::glue::<Tiny, _, (ByValue<i64>,), Val<i64>>(|_, (a,)| a)
                .at(&acvus_extern::CallSite::of_args(&site.args(1))),
            Width {
                args: 1,
                ret: 1,
                result: FormKind::Value,
                absent: false,
            },
            vec![erased(1i64)],
        ),
        1
    );
    assert_eq!(
        answered(
            acvus_extern::glue::<Tiny, _, (ByValue<i64>, ByValue<i64>), Val<i64>>(
                |_, (a, b)| a + b
            )
            .at(&acvus_extern::CallSite::of_args(&site.args(2))),
            Width {
                args: 2,
                ret: 1,
                result: FormKind::Value,
                absent: false,
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
            acvus_extern::glue::<
                Tiny,
                _,
                (ByValue<i64>, ByRef<i64, Shared>, ByValue<i64>),
                Val<i64>,
            >(|_, (a, b, c)| a + *b + c)
            .at(&acvus_extern::CallSite::of_args(&site.args(3))),
            Width {
                args: 3,
                ret: 1,
                result: FormKind::Value,
                absent: false,
            },
            vec![erased(1i64), lent, erased(3i64)],
        ),
        14,
        "a borrowed parameter reads the place the reference names"
    );

    assert_eq!(
        answered(
            acvus_extern::glue::<
                Tiny,
                _,
                (ByValue<i64>, ByValue<i64>, ByValue<i64>, ByValue<i64>,),
                Val<i64>,
            >(|_, (a, b, c, d)| a + b + c + d)
            .at(&acvus_extern::CallSite::of_args(&site.args(4))),
            Width {
                args: 4,
                ret: 1,
                result: FormKind::Value,
                absent: false,
            },
            vec![erased(1i64), erased(2i64), erased(3i64), erased(4i64)],
        ),
        10
    );

    let two_wide =
        acvus_extern::glue::<Tiny, _, (ByValue<i64>, ByValue<i64>), Val<i64>>(|_, (a, b)| {
            a * 10 + b
        })
        .at(&acvus_extern::CallSite::of_args(&site.args(2)));
    // SAFETY: the width says two arguments in and one value out.
    let by_register = unsafe {
        one_of(
            &two_wide,
            &mut tiny_ctx(),
            &[erased(1i64), erased(2i64)],
        )
    };
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
    use acvus_extern::{ByValue, HandlerFactory, Val, Width};

    let glue = acvus_extern::glue::<Tiny, _, (ByValue<i64>,), Val<i64>>(|_, (a,)| a * 3);
    let boxed: Box<dyn HandlerFactory<Tiny>> = Box::new(glue.clone());
    let again = boxed.clone();

    assert_eq!(
        boxed.width(),
        Width {
            args: 1,
            ret: 1,
            result: FormKind::Value,
            absent: false,
        }
    );
    assert_eq!(
        again.width(),
        Width {
            args: 1,
            ret: 1,
            result: FormKind::Value,
            absent: false,
        }
    );
    let site = acvus_extern::SitesNoParameterReads::default();
    let glue = glue.at(&acvus_extern::CallSite::of_args(&site.args(1)));
    // SAFETY: the width says one argument in and one value out, at each of
    // the three names of this one handler.
    let answers = unsafe {
        [
            one_of(&glue, &mut tiny_ctx(), &[erased(7i64)]),
            boxed
                .at_site(&acvus_extern::CallSite::of_args(&site.args(1)))
                .into_op(())
                .call_run(&Tiny, &[erased(7i64)]),
            again
                .at_site(&acvus_extern::CallSite::of_args(&site.args(1)))
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
            absent: false,
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
    let closure = |_: &mut Ctx<'_, Tiny>, (a, b): (i64, i64)| a + b;
    let unsited = acvus_extern::glue::<Tiny, _, (ByValue<i64>, ByValue<i64>), Val<i64>>(closure);
    let sited = unsited.at(&acvus_extern::CallSite::of_args(&site.args(2)));

    assert_eq!(size_of_val(&closure), 0);
    assert_eq!(size_of_val(&sited), 0);
}

// -- A family member is a Rust type written out (RFC-0041) ---------------

/// `Vec`'s two family casts at a member whose `#` argument has a part the
/// runtime fills: `Vec<#(#Float, T)>`, which `#[extern_fn]` builds for a
/// `Monomorphize` member beside a variable. Built by hand: a `Vec` of
/// `(f64, Owned<R>)` or of `(f64, Erased<R, i64>)` does not cross the
/// boundary, so `#[extern_fn]` does not build one today.
fn a_family_member_with_a_uniform_part() -> Registry<Tiny> {
    Registry::new(|i: &Interner| {
        let vec_of = |arg: TypeArg<acvus_extern::Poly>| PolyTy::UserDefined {
            id: acvus_extern::QualifiedRef::root(i.intern("Vec")),
            type_args: vec![arg],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        let t = PolyTy::Var(0);
        let handler = || {
            ExternHandler::sync(acvus_extern::glue::<Tiny, _, (), acvus_extern::Val<V>>(
                |_, ()| V::Taken,
            ))
        };
        let mut contribution = acvus_extern::Contribution::of(acvus_extern::Manifest {
            types: vec![acvus_extern::DeclaredType::of::<Vec<i64>>(i)],
            signatures: Vec::new(),
            fns: Vec::new(),
        });
        for cast in acvus_extern::family_casts::<Tiny>(
            i,
            acvus_extern::MemberType {
                specialized: vec_of(TypeArg::Specialized(acvus_extern::HeldTy::Tuple(vec![
                    TypeArg::specialized(PolyTy::Float),
                    TypeArg::uniform(t.clone()),
                ]))),
                uniform: vec_of(TypeArg::uniform(PolyTy::Tuple(vec![PolyTy::Float, t]))),
                erase: handler(),
                materialize: handler(),
            },
        ) {
            contribution.declare(cast);
        }
        contribution
    })
}

#[test]
fn a_family_member_with_a_part_the_runtime_fills_is_refused_where_the_registry_is_built() {
    let i = Interner::new();
    let err = Externs::combine(vec![a_family_member_with_a_uniform_part()], &i)
        .err()
        .expect("the pattern `Vec<#T>` would call the uniform part `#`");
    let acvus_extern::CombineError::FamilyMemberNotWritten { function, member } = &err else {
        panic!("{err:?}")
    };
    assert_eq!(function, "Vec::erase");
    assert_eq!(
        member.display(&i).to_string(),
        "Fn(Vec<#(#Float, T)>) -> Vec<(Float, T)>"
    );
}
