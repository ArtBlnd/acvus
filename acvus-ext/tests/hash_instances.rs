//! `#` instances made by `Monomorphize` (scenarios.md S8–S11): a member
//! instance's type carries `#` on the slots holding the member, its glue
//! crosses a `Vec<#f64>` as one box, and the macro declares the two casts
//! between the representations once per family. The numbers are counted
//! by `Counting`, a runtime whose `erase` and `materialize` count every
//! Rust value they box or unbox; a value that fits the runtime's word, as
//! `f64` and `i64` do in the interpreter, is neither.

use std::any::{Any, TypeId, type_name};
use std::future::Ready;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_ext::vec_registry;
use acvus_extern::{
    DirectOp, ExternHandler, Externs, FnKind, Interner, Monomorphize, OneValue, PolyTy,
    QualifiedRef, Registry, Release, Runtime, TyTerm, TypeArg, extern_fn, extern_registry,
};

/// No registry these tests combine declares a sliceable container, so the
/// pair a slice would occupy is never built or read.
const NO_SLICES: &str = "this runtime holds no slices";

// -- A counting runtime -----------------------------------------------

/// `Release: Copy` forbids a value that owns its payload inline, so an
/// owning shape here is one word naming a cell that `release` frees.
#[derive(Clone, Copy, Debug, Default)]
enum V {
    /// The value a handler took out of its argument slot.
    #[default]
    Taken,
    /// The language's `Option` (RFC-0039 rule 6), held as a host pleases.
    None,
    Some(*mut V),
    Word(*mut (dyn Any + Send + Sync)),
    Boxed(*mut (dyn Any + Send + Sync)),
    Reference(*const V),
}

// SAFETY: a cell is reached only through the value that owns it, and a
// `Reference` only while its target is live (RFC-0018).
unsafe impl Send for V {}
unsafe impl Sync for V {}

impl Release for V {
    fn release(self) {
        match self {
            V::Taken | V::None | V::Reference(_) => {}
            // SAFETY: `some` leaked this cell and nothing else releases it.
            V::Some(cell) => unsafe { *Box::from_raw(cell) }.release(),
            // SAFETY: `erase` leaked this cell and nothing else frees it.
            V::Word(cell) | V::Boxed(cell) => drop(unsafe { Box::from_raw(cell) }),
        }
    }
}

#[derive(Clone, Default)]
struct Counting {
    boxes: Arc<AtomicUsize>,
    unboxes: Arc<AtomicUsize>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Counts {
    boxes: usize,
    unboxes: usize,
}

impl Counting {
    fn counts(&self) -> Counts {
        Counts {
            boxes: self.boxes.load(Ordering::SeqCst),
            unboxes: self.unboxes.load(Ordering::SeqCst),
        }
    }

    fn since(&self, start: Counts) -> Counts {
        let now = self.counts();
        Counts {
            boxes: now.boxes - start.boxes,
            unboxes: now.unboxes - start.unboxes,
        }
    }
}

/// Whether the interpreter keeps a `T` in the value word itself: the
/// condition `acvus_extern::Inline` asserts of its types.
fn fits_the_word<T>() -> bool {
    std::mem::size_of::<T>() <= 8 && std::mem::align_of::<T>() <= 8 && !std::mem::needs_drop::<T>()
}

fn open_ref<T>(value: &V) -> &T
where
    T: Send + Sync + 'static,
{
    let (V::Boxed(cell) | V::Word(cell)) = value else {
        panic!("open_ref: not a value: {value:?}")
    };
    // SAFETY: the value is live, so its cell is.
    unsafe { &**cell }
        .downcast_ref::<T>()
        .unwrap_or_else(|| panic!("open_ref: value is not a {}", type_name::<T>()))
}

fn open_mut<T>(value: &mut V) -> &mut T
where
    T: Send + Sync + 'static,
{
    let (V::Boxed(cell) | V::Word(cell)) = value else {
        panic!("open_mut: not a value: {value:?}")
    };
    // SAFETY: `&mut V` is the exclusive name of the value and its cell.
    unsafe { &mut **cell }
        .downcast_mut::<T>()
        .unwrap_or_else(|| panic!("open_mut: value is not a {}", type_name::<T>()))
}

/// This runtime's registry declares no enum, so no value of it is ever a
/// variant and no tag register is ever written or read.
const NO_VARIANTS: &str = "this runtime holds no variants";

static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

acvus_extern::cross_one_value!(V, at Counting);

impl acvus_extern::OneValue<Counting> for V {
    fn erase(self, _: &Counting) -> V {
        self
    }

    unsafe fn materialize(_: &Counting, value: V) -> Self {
        value
    }
}

impl acvus_extern::Borrowable<Counting> for V {
    unsafe fn deref<'a>(_: &Counting, reference: &'a V) -> &'a V {
        let V::Reference(p) = reference else {
            panic!("deref: not a reference: {reference:?}")
        };
        // SAFETY: the target is live for as long as the reference is used.
        unsafe { &**p }
    }

    unsafe fn deref_mut<'a>(_: &Counting, reference: &'a V) -> &'a mut V {
        let V::Reference(p) = reference else {
            panic!("deref_mut: not a reference: {reference:?}")
        };
        // SAFETY: the target is live and, by the checker, exclusively named.
        unsafe { &mut *(*p as *mut V) }
    }
}

// SAFETY: `V` is this runtime's own value, which no Rust type disagrees with.
unsafe impl acvus_extern::FromValue<Counting> for V {
    unsafe fn from_value(_: &Counting, value: V) -> V {
        value
    }
}

impl Runtime for Counting {
    fn instance_value(_: &acvus_extern::InstanceEntry<Self>) -> Self::Value {
        panic!("Counting declares no instances")
    }

    unsafe fn instance_entry<'a>(_: &'a Self::Value) -> &'a acvus_extern::InstanceEntry<Self> {
        panic!("Counting declares no instances")
    }

    type Op = DirectOp<Counting>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = DirectOp<Counting>;
    type FusedShape = ();

    acvus_extern::direct_call_forms!();

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

    fn type_of(&self, value: &V) -> Option<TypeId> {
        let (V::Boxed(cell) | V::Word(cell)) = value else {
            return None;
        };
        // SAFETY: the value is live, so its cell is.
        Some(unsafe { &**cell }.type_id())
    }
    fn type_name_of(&self, _: &V) -> Option<&'static str> {
        None
    }

    unsafe fn materialize<T>(&self, value: V) -> T
    where
        T: Send + Sync + 'static,
    {
        if TypeId::of::<T>() == TypeId::of::<V>() {
            let boxed: Box<dyn Any> = Box::new(value);
            return *boxed.downcast::<T>().expect("T is V");
        }
        let cell = match value {
            V::Word(cell) => cell,
            V::Boxed(cell) => {
                self.unboxes.fetch_add(1, Ordering::SeqCst);
                cell
            }
            V::None | V::Some(_) | V::Reference(_) => {
                panic!("materialize: not a value: {value:?}")
            }
            V::Taken => panic!("materialize: the value was already taken out of its slot"),
        };
        // SAFETY: the caller's contract; the cell is taken, not released.
        let any = unsafe { Box::from_raw(cell) };
        match any.downcast::<T>() {
            Ok(v) => *v,
            Err(_) => panic!("materialize: value is not a {}", type_name::<T>()),
        }
    }

    unsafe fn erase<T>(&self, value: T) -> V
    where
        T: Send + Sync + 'static,
    {
        if TypeId::of::<T>() == TypeId::of::<V>() {
            let boxed: Box<dyn Any> = Box::new(value);
            return *boxed.downcast::<V>().expect("T is V");
        }
        let boxed: Box<dyn Any + Send + Sync> = Box::new(value);
        if fits_the_word::<T>() {
            return V::Word(Box::into_raw(boxed));
        }
        self.boxes.fetch_add(1, Ordering::SeqCst);
        V::Boxed(Box::into_raw(boxed))
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

    unsafe fn inline_ref<T>(value: &V) -> &T
    where
        T: acvus_extern::Inline,
    {
        open_ref(value)
    }

    unsafe fn inline_mut<T>(value: &mut V) -> &mut T
    where
        T: acvus_extern::Inline,
    {
        open_mut(value)
    }

    unsafe fn deref<'a, T>(&self, reference: &'a V) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        open_ref(unsafe { <V as acvus_extern::Borrowable<Counting>>::deref(self, reference) })
    }

    unsafe fn deref_mut<'a, T>(&self, reference: &'a V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        open_mut(unsafe { <V as acvus_extern::Borrowable<Counting>>::deref_mut(self, reference) })
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
        let V::Some(cell) = value else {
            panic!("unwrap_some: the value is not a Some")
        };
        // SAFETY: `some` leaked this cell; the payload moves out of it.
        *unsafe { Box::from_raw(cell) }
    }

    unsafe fn some_at<'a>(&self, value: &'a V) -> Option<&'a V> {
        let V::Some(cell) = value else {
            return None;
        };
        // SAFETY: `some` leaked this cell, and it lives as long as the
        // option that names it.
        Some(unsafe { &**cell })
    }

    unsafe fn some_at_mut<'a>(&self, value: &'a mut V) -> Option<&'a mut V> {
        let V::Some(cell) = value else {
            return None;
        };
        // SAFETY: as `some_at`, with the caller's exclusive loan.
        Some(unsafe { &mut **cell })
    }

    fn symbol(&self, name: &str) -> acvus_extern::Astr {
        SYMBOLS.intern(name)
    }

    fn variant_tag(&self, _: &str) -> Self::Value {
        panic!("{NO_VARIANTS}")
    }

    unsafe fn tag_symbol(&self, _: &Self::Value) -> acvus_extern::Astr {
        panic!("{NO_VARIANTS}")
    }

    fn undef(&self) -> Self::Value {
        panic!("{NO_VARIANTS}")
    }

    fn is_undef(&self, _: &Self::Value) -> bool {
        panic!("{NO_VARIANTS}")
    }

    fn slice_into_run(&self, _: acvus_extern::Words, _: &mut [Self::Value]) {
        panic!("{NO_SLICES}")
    }

    unsafe fn slice_from_run(&self, _: &[Self::Value]) -> acvus_extern::Words {
        panic!("{NO_SLICES}")
    }

    fn call_is_sync(&self, _: &V) -> bool {
        false
    }

    unsafe fn call_now<A>(&self, _: &V, _: &mut acvus_extern::Ctx<'_, Self>, _: A) -> V
    where
        A: acvus_extern::IntoRun<Self>,
    {
        self.no_closures()
    }

    unsafe fn call_0<'a>(&'a self, _: &'a V) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }

    unsafe fn call_1<'a>(&'a self, _: &'a V, _: V) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }

    unsafe fn call_n<'a>(&'a self, _: &'a V, _: &mut [V]) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }
}

// -- The member functions under test ------------------------------------

/// The arithmetic a member supplies; the bound keeps the generic instance
/// out, so `dot` and `zeros` exist only at their members.
trait Float: Copy + Send + Sync + 'static {
    const ZERO: Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl Float for f64 {
    const ZERO: Self = 0.0;

    fn mul_add(self, a: Self, b: Self) -> Self {
        self + a * b
    }
}

#[extern_fn(effect = pure)]
fn dot<T>(a: Vec<T>, b: Vec<T>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    a.iter()
        .zip(&b)
        .fold(T::ZERO, |acc, (x, y)| acc.mul_add(*x, *y))
}

#[extern_fn(effect = pure)]
fn zeros<T>(n: i64) -> Vec<T>
where
    T: Monomorphize<(f64,)> + Float,
{
    vec![T::ZERO; usize::try_from(n).expect("a count is not negative")]
}

fn registries() -> Vec<Registry<Counting>> {
    vec![
        vec_registry(),
        extern_registry! {
            ns: "t",
            fns: [dot, zeros],
        },
    ]
}

struct World {
    rt: Counting,
    externs: Externs<Counting>,
    interner: Interner,
}

impl World {
    fn new() -> Self {
        let interner = Interner::new();
        let externs = Externs::combine(registries(), &interner).expect("registries combine");
        Self {
            rt: Counting::default(),
            externs,
            interner,
        }
    }

    fn qref(&self, ns: &str, name: &str) -> QualifiedRef {
        QualifiedRef::qualified(self.interner.intern(ns), self.interner.intern(name))
    }

    /// Calls instance `instance` of `ns::name`, as the compiler's
    /// `Callee::Extern` numbers them (RFC-0040).
    fn call(&self, ns: &str, name: &str, instance: usize, args: Vec<V>) -> V {
        let handlers = &self.externs.handlers[&self.qref(ns, name)];
        let ExternHandler::Sync(handler) = &handlers[instance] else {
            panic!("{ns}::{name} is not a sync handler")
        };
        let site = acvus_extern::SitesNoParameterReads::default();
        let op = handler
            .clone()
            .at_site(&acvus_extern::CallSite::of_args(
                &site.args(handler.arity()),
            ))
            .into_op(());
        // SAFETY: the caller passes the declaration's own arguments.
        unsafe { op.call_run(&self.rt, &args) }
    }

    fn function(&self, ns: &str, name: &str) -> &acvus_extern::Function {
        let qref = self.qref(ns, name);
        self.externs
            .functions
            .iter()
            .find(|f| f.qref == qref)
            .unwrap_or_else(|| panic!("{ns}::{name} is declared"))
    }

    fn instances(&self, ns: &str, name: &str) -> &acvus_mir::ty::Instances {
        let FnKind::Extern { instances, .. } = &self.function(ns, name).kind else {
            panic!("{ns}::{name} is extern")
        };
        instances
    }

    fn specialized_floats(&self, items: &[f64]) -> V {
        // SAFETY: a `#` `Vec<f64>` is the Rust `Vec<f64>` in one box.
        unsafe { self.rt.erase::<Vec<f64>>(items.to_vec()) }
    }

    fn uniform_floats(&self, items: &[f64]) -> V {
        let words: Vec<V> = items
            .iter()
            // SAFETY: stored as itself.
            .map(|x| unsafe { self.rt.erase::<f64>(*x) })
            .collect();
        acvus_extern::OneValue::<_>::erase(words, &self.rt)
    }

    fn float(&self, value: V) -> f64 {
        // SAFETY: the callee returned an `f64`.
        unsafe { self.rt.materialize::<f64>(value) }
    }

    fn word(&self, n: i64) -> V {
        // SAFETY: stored as itself.
        unsafe { self.rt.erase::<i64>(n) }
    }
}

fn vec_of(i: &Interner, arg: TypeArg<acvus_extern::Poly>) -> PolyTy {
    PolyTy::UserDefined {
        id: QualifiedRef::root(i.intern("Vec")),
        type_args: vec![arg],
        effect_args: vec![],
        identity_args: vec![],
        region_params: 0,
    }
}

// -- Signatures ---------------------------------------------------------

#[test]
fn a_member_instance_marks_the_slots_holding_the_member() {
    let w = World::new();
    let instances = w.instances("t", "dot");
    assert!(
        !instances.generic,
        "dot has no generic instance: `Float` is not on the value"
    );
    let [at_f64] = instances.concrete.as_slice() else {
        panic!("dot has one member")
    };
    let TyTerm::Fn { params, ret, .. } = &at_f64.ty else {
        panic!("an instance is a function type")
    };
    let expected = vec_of(&w.interner, TypeArg::specialized(PolyTy::Float));
    assert_eq!(params[0].ty, expected);
    assert_eq!(params[1].ty, expected);
    assert_eq!(**ret, PolyTy::Float);
    assert_eq!(
        at_f64.ty.display(&w.interner).to_string(),
        "Fn(Vec<#Float>, Vec<#Float>) -> Float"
    );
}

#[test]
fn a_type_variable_at_a_specializing_slot_carries_its_slot_s_rho() {
    let w = World::new();
    let TyTerm::Fn { ret, .. } = &w.function("t", "zeros").ty else {
        panic!("zeros has a function type")
    };
    let TyTerm::UserDefined { type_args, .. } = &**ret else {
        panic!("zeros returns a Vec")
    };
    assert!(matches!(type_args[0], TypeArg::Open(_, TyTerm::Var(_))));
}

// -- Cast rules ---------------------------------------------------------

#[test]
fn a_family_in_a_member_signature_declares_its_two_casts_once() {
    let w = World::new();
    let vec = QualifiedRef::root(w.interner.intern("Vec"));
    let mut rules: Vec<(String, String, String)> = w
        .externs
        .types
        .rules_from(vec)
        .iter()
        .map(|r| {
            (
                w.interner.resolve(r.fn_ref.name).to_string(),
                r.from.display(&w.interner).to_string(),
                r.to.display(&w.interner).to_string(),
            )
        })
        .collect();
    rules.sort();
    assert_eq!(
        rules,
        vec![
            (
                "erase".to_string(),
                "Vec<#T>".to_string(),
                "Vec<T>".to_string()
            ),
            (
                "materialize".to_string(),
                "Vec<T>".to_string(),
                "Vec<#T>".to_string()
            ),
        ]
    );
    for name in ["erase", "materialize"] {
        let instances = w.instances("Vec", name);
        assert!(!instances.generic);
        assert_eq!(
            instances.concrete.len(),
            1,
            "dot and zeros declare one {name} at f64"
        );
    }
    assert_eq!(
        w.instances("Vec", "materialize").concrete[0]
            .ty
            .display(&w.interner)
            .to_string(),
        "Fn(Vec<Float>) -> Vec<#Float>"
    );
}

// -- S8 -----------------------------------------------------------------

#[test]
fn s8_dot_on_two_specialized_vecs_unboxes_each_once_and_boxes_nothing() {
    let w = World::new();
    let a = w.specialized_floats(&[1.0, 2.0, 3.0]);
    let b = w.specialized_floats(&[4.0, 5.0, 6.0]);
    let start = w.rt.counts();
    let r = w.call("t", "dot", 0, vec![a, b]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 0,
            unboxes: 2
        }
    );
    assert_eq!(w.float(r), 32.0);
}

// -- S9 -----------------------------------------------------------------

#[test]
fn s9_a_uniform_argument_is_materialized_once_then_dot_runs_as_s8() {
    let w = World::new();
    let a = w.uniform_floats(&[1.0, 2.0, 3.0]);
    let b = w.uniform_floats(&[4.0, 5.0, 6.0]);
    let start = w.rt.counts();
    let a = w.call("Vec", "materialize", 0, vec![a]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 1,
            unboxes: 1
        }
    );
    let b = w.call("Vec", "materialize", 0, vec![b]);
    let r = w.call("t", "dot", 0, vec![a, b]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 2,
            unboxes: 4
        }
    );
    assert_eq!(w.float(r), 32.0);
}

// -- S10 ----------------------------------------------------------------

#[test]
fn s10_a_specialized_result_is_erased_once_for_a_generic_consumer() {
    let w = World::new();
    let start = w.rt.counts();
    let z = w.call("t", "zeros", 0, vec![w.word(3)]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 1,
            unboxes: 0
        }
    );
    let start = w.rt.counts();
    let z = w.call("Vec", "erase", 0, vec![z]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 1,
            unboxes: 1
        }
    );
    let start = w.rt.counts();
    let reversed = w.call("vec", "reverse", 0, vec![z]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 1,
            unboxes: 1
        }
    );
    // SAFETY: `reverse` returns the uniform `Vec<V>` it was given.
    let items = unsafe { <Vec<V> as OneValue<Counting>>::materialize(&w.rt, reversed) };
    assert_eq!(items.len(), 3);
}

// -- S11 ----------------------------------------------------------------

#[test]
fn s11_a_specialized_producer_feeding_a_specialized_consumer_converts_nothing() {
    let w = World::new();
    let start = w.rt.counts();
    let a = w.call("t", "zeros", 0, vec![w.word(3)]);
    let b = w.call("t", "zeros", 0, vec![w.word(3)]);
    let r = w.call("t", "dot", 0, vec![a, b]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 2,
            unboxes: 2
        }
    );
    assert_eq!(w.float(r), 0.0);
}

impl Counting {
    fn no_closures(&self) -> V {
        panic!("Counting runs no closures")
    }
}
