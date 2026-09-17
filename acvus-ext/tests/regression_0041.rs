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
    Arr, CallToken, Erased, Externs, Fn1, FnKind, FromValue, Interner, Monomorphize, QualifiedRef,
    Ref, RefMut, Registry, Runtime, Trap, extern_fn, extern_registry,
};

// -- A counting runtime -----------------------------------------------

#[derive(Debug, Default)]
enum V {
    /// The value a handler took out of its argument slot.
    #[default]
    Taken,
    Boxed(Box<dyn Any + Send + Sync>),
    Reference(*const V),
}

// SAFETY: a `Reference` is used only while its target is live (RFC-0018).
unsafe impl Send for V {}
unsafe impl Sync for V {}

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
    let V::Boxed(any) = value else {
        panic!("open_ref: not a value: {value:?}")
    };
    any.downcast_ref::<T>()
        .unwrap_or_else(|| panic!("open_ref: value is not a {}", type_name::<T>()))
}

fn open_mut<T>(value: &mut V) -> &mut T
where
    T: Send + Sync + 'static,
{
    let V::Boxed(any) = value else {
        panic!("open_mut: not a value: {value:?}")
    };
    any.downcast_mut::<T>()
        .unwrap_or_else(|| panic!("open_mut: value is not a {}", type_name::<T>()))
}

static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

impl acvus_extern::Cross<Counting> for V {
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
    fn from_value(_: &Counting, value: V) -> Result<V, Trap> {
        Ok(value)
    }
}

impl Runtime for Counting {
    type Value = V;
    type Error = Trap;
    type CallFuture<'a> = Ready<Result<V, Trap>>;

    fn type_of(&self, value: &V) -> Option<TypeId> {
        let V::Boxed(any) = value else {
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
        let V::Boxed(any) = value else {
            panic!("materialize: not a value: {value:?}")
        };
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
        V::Boxed(Box::new(value))
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
        open_ref(unsafe { <V as acvus_extern::Cross<Counting>>::deref(self, reference) })
    }

    unsafe fn deref_mut<'a, T>(&self, reference: &'a V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        open_mut(unsafe { <V as acvus_extern::Cross<Counting>>::deref_mut(self, reference) })
    }

    unsafe fn reference(&self, target: &V) -> V {
        V::Reference(target as *const V)
    }

    fn symbol(&self, name: &str) -> acvus_extern::Astr {
        SYMBOLS.intern(name)
    }

    fn call_is_sync(&self, _: &V) -> bool {
        true
    }

    fn call_now(&self, f: &V, args: &mut [V], _: CallToken) -> Result<V, Trap> {
        let [a] = args else {
            return Err(Trap::internal("Counting runs only unary closures"));
        };
        Ok(open_ref::<UnaryClosure>(f)(self, std::mem::take(a)))
    }

    fn call_0<'a>(&'a self, _: &'a V, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(Err(Trap::internal("Counting runs only unary closures")))
    }

    fn call_1<'a>(&'a self, f: &'a V, a: V, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(Ok(open_ref::<UnaryClosure>(f)(self, a)))
    }

    fn call_n<'a>(&'a self, _: &'a V, _: &mut [V], _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(Err(Trap::internal("Counting runs only unary closures")))
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

type It = Iter<V, (), (), Counting>;

fn drain(rt: &Counting, mut it: It) -> Vec<i64> {
    futures::executor::block_on(async {
        let mut out = Vec::new();
        while let Some(value) = it.next_value(rt).await.expect("a stage yields") {
            out.push(read_int(rt, &value));
        }
        out
    })
}

fn items(rt: &Counting, ns: impl IntoIterator<Item = i64>) -> It {
    Iter::from_items(ns.into_iter().map(|n| int(rt, n)).collect())
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
        let back = Erased::<Counting, $t>::from_value(&$rt, raw)
            .unwrap_or_else(|e| panic!("{} is read back as itself: {e}", type_name::<$t>()));
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
fn from_value_on_a_bool_as_an_i64_traps_naming_both() {
    let rt = Counting::default();
    let holds_a_bool = erased_from(&rt, true);
    let Err(trap) = Erased::<Counting, i64>::from_value(&rt, holds_a_bool) else {
        panic!("a bool is read back as an i64")
    };
    let message = trap.to_string();
    assert!(
        message.contains(type_name::<i64>()),
        "the trap names the expected type: {message}"
    );
    assert!(
        message.contains(&format!("{:?}", TypeId::of::<bool>())),
        "the trap names the found type by its TypeId, this runtime keeping no name: {message}"
    );
}

// -- R3: container downcasts ---------------------------------------------

#[test]
fn vec_from_value_refuses_a_deque_and_takes_a_vec_of_values_with_no_per_element_unbox() {
    let rt = Counting::default();
    let deque = erased_from(&rt, Deque::<V>::default());
    let Err(trap) = Vec::<Erased<Counting, String>>::from_value(&rt, deque) else {
        panic!("a Deque is read back as a Vec")
    };
    assert!(trap.to_string().contains("Vec"), "{trap}");

    let strings = erased_from(
        &rt,
        vec![
            erased_from(&rt, "a".to_owned()),
            erased_from(&rt, "b".to_owned()),
        ],
    );
    let start = rt.counts();
    let parts = Vec::<Erased<Counting, String>>::from_value(&rt, strings).expect("a Vec of String");
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
fn arr_from_value_refuses_a_deque_and_takes_an_array_of_values_with_no_per_element_unbox() {
    let rt = Counting::default();
    let deque = erased_from(&rt, Deque::<V>::default());
    let Err(trap) = Arr::<Erased<Counting, String>, ()>::from_value(&rt, deque) else {
        panic!("a Deque is read back as an Arr")
    };
    assert!(trap.to_string().contains("Arr"), "{trap}");

    let strings = erased_from(
        &rt,
        Arr::<V, ()>::new(vec![
            erased_from(&rt, "a".to_owned()),
            erased_from(&rt, "b".to_owned()),
        ]),
    );
    let start = rt.counts();
    let parts = Arr::<Erased<Counting, String>, ()>::from_value(&rt, strings).expect("an Arr");
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
    let storage = erased_from(&rt, vec![int(&rt, 1), int(&rt, 2)]);
    let start = rt.counts();

    let lent = Ref::<Vec<Erased<Counting, i64>>, Counting>::lend(&rt, &storage);
    let seen: Vec<i64> = lent.as_slice(&rt).iter().map(|x| x.get()).collect();
    assert_eq!(seen, [1, 2]);

    // SAFETY: `storage` is live and unmoved; no other name reads it during
    // the edit.
    let mut lent_mut =
        RefMut::<Vec<Erased<Counting, i64>>, Counting>::new(unsafe { rt.reference(&storage) });
    for x in lent_mut.as_mut_slice(&rt) {
        **x += 1;
    }

    let again = Ref::<Vec<Erased<Counting, i64>>, Counting>::lend(&rt, &storage);
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
    let it = items(&rt, [1, 2, 3]).map::<V>(Fn1::new(&rt, f)).take(2);
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
        .filter(Fn1::new(&rt, keep_odd))
        .map::<V>(Fn1::new(&rt, times_ten));
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
        let inner: Vec<V> = if x == 2 {
            vec![]
        } else {
            vec![int(rt, x), int(rt, x)]
        };
        erased_from(rt, inner)
    });
    let it = items(&rt, [1, 2, 3]).flat_map::<Vec<V>, V>(Fn1::new(&rt, twice_unless_two));
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
        at_f64.display(&interner).to_string(),
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
        .map(|t| t.display(&interner).to_string())
        .collect();
    assert_eq!(
        displayed,
        ["Fn(Vec<Option<Float>>) -> Vec<#Option<Float>>"],
        "the one cast instance is at the composite Option<Float>"
    );
}
