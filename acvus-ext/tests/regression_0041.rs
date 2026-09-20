//! Regression tests for RFC-0041 at the extension boundary (R1–R6): every
//! `Inline` type round-trips through `Erased`, the checked exit refuses
//! the wrong type by name, container downcasts count no per-element
//! unbox, a `Ref` reads and edits elements in place, an `Iter` pipeline is
//! lazy, and a `Monomorphize` member whose slot holds a composite has the
//! signature and casts the registry gives it. The numbers are counted by
//! `Counting`, the runtime of `erased.rs`, extended to run a Rust closure
//! held in a value so `map`/`filter` can be driven.

use std::any::{Any, TypeId, type_name};
use std::future::Ready;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use acvus_ext::{Deque, Iter, vec_registry};
use acvus_extern::{
    Arr, Closure, Erased, Externs, FnKind, FromValue, Interner, Monomorphize, Mut, OneValue, Owned,
    QualifiedRef, Ref, Registry, Release, Runtime, Shared, extern_fn, extern_registry,
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
    /// The language's `Option` (RFC-0022), held as a host pleases.
    None,
    Some(*mut V),
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
            V::Boxed(cell) => drop(unsafe { Box::from_raw(cell) }),
        }
    }
}

type UnaryClosure = Box<dyn Fn(&Counting, V) -> V + Send + Sync>;

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

fn open_ref<T>(value: &V) -> &T
where
    T: Send + Sync + 'static,
{
    let V::Boxed(cell) = value else {
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
    let V::Boxed(cell) = value else {
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

impl acvus_extern::FromValue<Counting> for V {
    fn from_value(_: &Counting, value: V) -> V {
        value
    }
}

impl Runtime for Counting {
    type Op = acvus_extern::DirectOp<Counting>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = acvus_extern::DirectOp<Counting>;
    type FusedShape = ();

    acvus_extern::direct_call_forms!();

    type Value = V;
    type Frame<'a> = ();
    type Rooted = ();
    type CallFuture<'a> = Ready<V>;

    fn rooted(&self) {}
    fn frame_of(_: &mut ()) {}

    fn type_of(&self, value: &V) -> Option<TypeId> {
        let V::Boxed(cell) = value else {
            return None;
        };
        // SAFETY: the value is live, so its cell is.
        Some(unsafe { &**cell }.type_id())
    }
    fn type_name_of(&self, _: &V) -> Option<&'static str> {
        None
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

    unsafe fn materialize<T>(&self, value: V) -> T
    where
        T: Send + Sync + 'static,
    {
        if TypeId::of::<T>() == TypeId::of::<V>() {
            let boxed: Box<dyn Any> = Box::new(value);
            return *boxed.downcast::<T>().expect("T is V");
        }
        self.unboxes.fetch_add(1, Ordering::SeqCst);
        let V::Boxed(cell) = value else {
            panic!("materialize: not a value: {value:?}")
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
        self.boxes.fetch_add(1, Ordering::SeqCst);
        let boxed: Box<dyn Any + Send + Sync> = Box::new(value);
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

    unsafe fn deref<'a, T>(&self, reference: &'a V) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        open_ref(unsafe { <V as acvus_extern::OneValue<Counting>>::deref(self, reference) })
    }

    unsafe fn deref_mut<'a, T>(&self, reference: &'a V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        open_mut(unsafe { <V as acvus_extern::OneValue<Counting>>::deref_mut(self, reference) })
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
        true
    }

    unsafe fn call_now<A>(&self, f: &V, _: &mut (), args: A) -> V
    where
        A: acvus_extern::IntoRun<Self>,
    {
        let run = run_of(self, args);
        let [a] = run.as_slice() else {
            return self.only_unary();
        };
        open_ref::<UnaryClosure>(f)(self, *a)
    }

    unsafe fn call_0<'a>(&'a self, _: &'a V) -> Self::CallFuture<'a> {
        std::future::ready(self.only_unary())
    }

    unsafe fn call_1<'a>(&'a self, f: &'a V, a: V) -> Self::CallFuture<'a> {
        std::future::ready(open_ref::<UnaryClosure>(f)(self, a))
    }

    unsafe fn call_n<'a>(&'a self, _: &'a V, _: &mut [V]) -> Self::CallFuture<'a> {
        std::future::ready(self.only_unary())
    }
}

impl Counting {
    fn only_unary(&self) -> V {
        panic!("Counting runs only unary closures")
    }
}

fn erased_from<T>(rt: &Counting, value: T) -> V
where
    T: Send + Sync + 'static,
{
    // SAFETY: stored as itself.
    unsafe { rt.erase::<T>(value) }
}

fn int(rt: &Counting, n: i64) -> V {
    erased_from(rt, n)
}

fn read_int(rt: &Counting, value: &V) -> i64 {
    // SAFETY: the value was erased from an `i64`.
    *unsafe { rt.value_as_ref::<i64>(value) }
}

fn closure(rt: &Counting, f: impl Fn(&Counting, V) -> V + Send + Sync + 'static) -> V {
    let f: UnaryClosure = Box::new(f);
    erased_from(rt, f)
}

type It = Iter<Owned<Counting>, (), (), Counting>;

fn drain(rt: &Counting, mut it: It) -> Vec<i64> {
    futures::executor::block_on(async {
        let mut out = Vec::new();
        while let Some(value) = it.next_value(rt, &mut ()).await {
            out.push(read_int(rt, &value));
        }
        out
    })
}

fn items(rt: &Counting, ns: impl IntoIterator<Item = i64>) -> It {
    Iter::from_items(
        ns.into_iter()
            .map(|n| Owned::from_value(int(rt, n)))
            .collect(),
    )
}

// -- R1: every Inline type round-trips through Erased -------------------

macro_rules! inline_round_trip {
    ($rt:expr; $($t:ty = $v:expr),* $(,)?) => { $( {
        let value: $t = $v;
        let erased = Erased::<Counting, $t>::new(&$rt, value);
        assert_eq!(erased.get(), value, "{} reads back by Deref", type_name::<$t>());
        let raw = erased.into_value();
        assert_eq!(
            $rt.type_of(&raw),
            Some(TypeId::of::<$t>()),
            "{} records its TypeId",
            type_name::<$t>()
        );
        let back = Erased::<Counting, $t>::from_value(&$rt, raw);
        assert_eq!(back.into_inner(&$rt), value, "{} materializes", type_name::<$t>());
    } )* };
}

#[test]
fn every_inline_type_erased_records_its_type_id_and_materializes_back() {
    let rt = Counting::default();
    inline_round_trip!(rt;
        i8 = -8, i16 = -16, i32 = -32, i64 = -64,
        u8 = 8, u16 = 16, u32 = 32, u64 = 64,
        f64 = 1.5, bool = true, () = (),
    );
    assert_eq!(
        rt.counts(),
        Counts {
            boxes: 11,
            unboxes: 11
        },
        "one box per `new` and one unbox per `into_inner`"
    );
}

// -- R2: the checked exit names both types -------------------------------

#[test]
#[should_panic(expected = "expected a value erased from `i64`, found a payload of TypeId")]
fn from_value_on_a_bool_as_an_i64_panics_naming_both() {
    let rt = Counting::default();
    let holds_a_bool = erased_from(&rt, true);
    Erased::<Counting, i64>::from_value(&rt, holds_a_bool);
}

// -- R3: container downcasts ---------------------------------------------

#[test]
#[should_panic(expected = "expected a value erased from `alloc::vec::Vec<")]
fn vec_from_value_refuses_a_deque() {
    let rt = Counting::default();
    let deque = erased_from(&rt, Deque::<Owned<Counting>>::default());
    Vec::<Erased<Counting, String>>::from_value(&rt, deque);
}

#[test]
fn vec_from_value_takes_a_vec_of_values_with_no_per_element_unbox() {
    let rt = Counting::default();
    let strings = OneValue::<_>::erase(
        vec![
            erased_from(&rt, "a".to_owned()),
            erased_from(&rt, "b".to_owned()),
        ],
        &rt,
    );
    let start = rt.counts();
    let parts = Vec::<Erased<Counting, String>>::from_value(&rt, strings);
    assert_eq!(
        rt.since(start),
        Counts {
            boxes: 0,
            unboxes: 1
        },
        "the Vec box is opened once and no element is unboxed"
    );
    let read: Vec<&str> = parts.iter().map(|s| s.as_ref(&rt).as_str()).collect();
    assert_eq!(read, ["a", "b"]);
}

#[test]
#[should_panic(expected = "expected a value erased from `acvus_extern::len::Arr<")]
fn arr_from_value_refuses_a_deque() {
    let rt = Counting::default();
    let deque = erased_from(&rt, Deque::<Owned<Counting>>::default());
    Arr::<Erased<Counting, String>, ()>::from_value(&rt, deque);
}

#[test]
fn arr_from_value_takes_an_array_of_values_with_no_per_element_unbox() {
    let rt = Counting::default();
    let strings = OneValue::<Counting>::erase(
        Arr::<Owned<Counting>, ()>::new(vec![
            Owned::from_value(erased_from(&rt, "a".to_owned())),
            Owned::from_value(erased_from(&rt, "b".to_owned())),
        ]),
        &rt,
    );
    let start = rt.counts();
    let parts = Arr::<Erased<Counting, String>, ()>::from_value(&rt, strings);
    assert_eq!(
        rt.since(start),
        Counts {
            boxes: 0,
            unboxes: 1
        },
        "the Arr box is opened once and no element is unboxed"
    );
    let read: Vec<&str> = parts.0.iter().map(|s| s.as_ref(&rt).as_str()).collect();
    assert_eq!(read, ["a", "b"]);
}

// -- R4: a Ref reads and edits elements in place ---------------------------

#[test]
fn a_ref_to_a_vec_of_erased_ints_sees_the_elements_and_an_edit_through_a_mutable_one() {
    let rt = Counting::default();
    let storage = OneValue::<_>::erase(vec![int(&rt, 1), int(&rt, 2)], &rt);
    let start = rt.counts();

    let lent = Ref::<Vec<Erased<Counting, i64>>, Shared, Counting>::lend(&rt, &storage);
    let seen: Vec<i64> = lent.as_slice(&rt).iter().map(|x| x.get()).collect();
    assert_eq!(seen, [1, 2]);

    // SAFETY: `storage` is live and unmoved; no other name reads it during
    // the edit.
    let lent_mut =
        Ref::<Vec<Erased<Counting, i64>>, Mut, Counting>::new(unsafe { rt.reference(&storage) });
    for x in lent_mut.as_slice(&rt) {
        **x += 1;
    }

    let again = Ref::<Vec<Erased<Counting, i64>>, Shared, Counting>::lend(&rt, &storage);
    let seen: Vec<i64> = again.as_slice(&rt).iter().map(|x| x.get()).collect();
    assert_eq!(seen, [2, 3]);
    assert_eq!(
        rt.since(start),
        Counts {
            boxes: 0,
            unboxes: 0
        },
        "reading and editing in place boxes and unboxes nothing"
    );
}

// -- R5: an Iter pipeline is lazy ------------------------------------------

#[test]
fn map_then_take_two_calls_the_closure_exactly_twice() {
    let rt = Counting::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let f = {
        let calls = calls.clone();
        closure(&rt, move |rt, x| {
            calls.fetch_add(1, Ordering::SeqCst);
            int(rt, read_int(rt, &x) * 10)
        })
    };
    let it = items(&rt, [1, 2, 3])
        .map::<Owned<Counting>>(Closure::new(&rt, f))
        .take(2);
    assert_eq!(drain(&rt, it), [10, 20]);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn filter_then_map_interleave_per_element() {
    let rt = Counting::default();
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let keep_odd = {
        let log = log.clone();
        closure(&rt, move |rt, r| {
            // SAFETY: the filter lends a live element.
            let x = *unsafe { rt.deref::<i64>(&r) };
            log.lock().unwrap().push(format!("f{x}"));
            erased_from(rt, x % 2 == 1)
        })
    };
    let times_ten = {
        let log = log.clone();
        closure(&rt, move |rt, x| {
            let x = read_int(rt, &x);
            log.lock().unwrap().push(format!("m{x}"));
            int(rt, x * 10)
        })
    };
    let it = items(&rt, [1, 2, 3])
        .filter(Closure::new(&rt, keep_odd))
        .map::<Owned<Counting>>(Closure::new(&rt, times_ten));
    assert_eq!(drain(&rt, it), [10, 30]);
    assert_eq!(
        *log.lock().unwrap(),
        ["f1", "m1", "f2", "f3", "m3"],
        "a lazy pipeline maps an element before filtering the next"
    );
}

#[test]
fn flat_map_skips_an_empty_inner_sequence() {
    let rt = Counting::default();
    let twice_unless_two = closure(&rt, |rt, x| {
        let x = read_int(rt, &x);
        let inner: Vec<Owned<Counting>> = if x == 2 {
            vec![]
        } else {
            vec![Owned::from_value(int(rt, x)), Owned::from_value(int(rt, x))]
        };
        OneValue::<_>::erase(inner, rt)
    });
    let it = items(&rt, [1, 2, 3])
        .flat_map::<Vec<Owned<Counting>>, Owned<Counting>>(Closure::new(&rt, twice_unless_two));
    assert_eq!(drain(&rt, it), [1, 1, 3, 3]);
}

#[test]
fn chain_of_three_and_empty_and_empty_and_three_yields_six_in_order() {
    let rt = Counting::default();
    let it: It = items(&rt, [1, 2, 3])
        .chain::<(), ()>(items(&rt, []))
        .chain::<(), ()>(items(&rt, []))
        .chain::<(), ()>(items(&rt, [4, 5, 6]));
    assert_eq!(drain(&rt, it), [1, 2, 3, 4, 5, 6]);
}

// -- R6: a member under a composite slot ------------------------------------

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DisplayedCast {
    name: String,
    from: String,
    to: String,
}

trait Float: Copy + Send + Sync + 'static {
    const ZERO: Self;
    fn add(self, other: Self) -> Self;
}

impl Float for f64 {
    const ZERO: Self = 0.0;

    fn add(self, other: Self) -> Self {
        self + other
    }
}

#[extern_fn(effect = pure)]
fn sum_opt<T>(v: Vec<Option<T>>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    v.into_iter().flatten().fold(T::ZERO, Float::add)
}

fn registries() -> Vec<Registry<Counting>> {
    vec![
        vec_registry(),
        extern_registry! {
            ns: "t",
            fns: [sum_opt],
        },
    ]
}

#[test]
fn a_member_under_an_option_inside_a_vec_marks_the_vec_slot_and_declares_the_vec_casts() {
    let interner = Interner::new();
    let externs = Externs::combine(registries(), &interner).expect("registries combine");
    let qref = QualifiedRef::qualified(interner.intern("t"), interner.intern("sum_opt"));
    let function = externs
        .functions
        .iter()
        .find(|f| f.qref == qref)
        .expect("sum_opt is declared");
    let FnKind::Extern { instances, .. } = &function.kind else {
        panic!("sum_opt is extern")
    };
    assert!(!instances.generic, "Float keeps the generic instance out");
    let [at_f64] = instances.concrete.as_slice() else {
        panic!("sum_opt has one member")
    };
    assert_eq!(
        at_f64.ty.display(&interner).to_string(),
        "Fn(Vec<#Option<Float>>) -> Float",
        "the `#` sits on the Vec slot, which holds the composite whole"
    );

    let vec = QualifiedRef::root(interner.intern("Vec"));
    let mut rules: Vec<DisplayedCast> = externs
        .types
        .rules_from(vec)
        .iter()
        .map(|r| DisplayedCast {
            name: interner.resolve(r.fn_ref.name).to_string(),
            from: r.from.display(&interner).to_string(),
            to: r.to.display(&interner).to_string(),
        })
        .collect();
    rules.sort();
    assert_eq!(
        rules,
        vec![
            DisplayedCast {
                name: "erase".to_string(),
                from: "Vec<#'0>".to_string(),
                to: "Vec<'0>".to_string(),
            },
            DisplayedCast {
                name: "materialize".to_string(),
                from: "Vec<'0>".to_string(),
                to: "Vec<#'0>".to_string(),
            },
        ],
        "the family Vec declares its two casts once"
    );
    let materialize_qref =
        QualifiedRef::qualified(interner.intern("Vec"), interner.intern("materialize"));
    let materialize = externs
        .functions
        .iter()
        .find(|f| f.qref == materialize_qref)
        .expect("Vec::materialize is declared");
    let FnKind::Extern { instances, .. } = &materialize.kind else {
        panic!("Vec::materialize is extern")
    };
    let displayed: Vec<String> = instances
        .concrete
        .iter()
        .map(|t| t.ty.display(&interner).to_string())
        .collect();
    assert_eq!(
        displayed,
        ["Fn(Vec<Option<Float>>) -> Vec<#Option<Float>>"],
        "the one cast instance is at the composite Option<Float>"
    );
}

/// The arguments of a closure call, read back as the run this runtime's own
/// `call_now` reads them from.
fn run_of<Rt, A>(rt: &Rt, args: A) -> Vec<Rt::Value>
where
    Rt: acvus_extern::Runtime,
    A: acvus_extern::IntoRun<Rt>,
{
    let mut run = vec![Rt::Value::default(); A::WIDTH];
    args.into_run(rt, &mut run);
    run
}
