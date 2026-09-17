//! `Erased<R, String>` at the boundary (E1): a `Vec` of it crosses as one
//! box, and a body reads its elements without taking any back. The
//! numbers are counted by `Counting`, a runtime whose `erase` and
//! `materialize` count every Rust value they box or unbox.

use std::any::{Any, TypeId, type_name};
use std::future::Ready;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_ext::{string_registry, vec_registry};
use acvus_extern::{
    CallToken, Erased, ExternHandler, Externs, FromValue, Interner, PolyTy, QualifiedRef, Ref,
    Registry, Runtime, TyTerm, TypeArg, extern_fn, extern_registry,
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
    fn from_value(_: &Counting, value: V) -> V {
        value
    }
}

impl Runtime for Counting {
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

    type Value = V;
    type CallFuture<'a> = Ready<V>;

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
        false
    }

    fn call_now(&self, _: &V, _: &mut [V], _: CallToken) -> V {
        self.no_closures()
    }

    fn call_0<'a>(&'a self, _: &'a V, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }

    fn call_1<'a>(&'a self, _: &'a V, _: V, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }

    fn call_n<'a>(&'a self, _: &'a V, _: &mut [V], _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }
}

// -- The reader under test --------------------------------------------

#[extern_fn(effect = pure)]
fn join_erased<Rt>(rt: &Rt, items: Ref<Vec<Erased<Rt, String>>, Rt>) -> String
where
    Rt: Runtime,
{
    let parts: Vec<&str> = items
        .as_slice(rt)
        .iter()
        .map(|s| s.as_ref(rt).as_str())
        .collect();
    parts.join(",")
}

fn registries() -> Vec<Registry<Counting>> {
    vec![
        string_registry(),
        vec_registry(),
        extern_registry! {
            ns: "t",
            fns: [join_erased],
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

    fn call(&self, ns: &str, name: &str, mut args: Vec<V>) -> V {
        let qref = QualifiedRef::qualified(self.interner.intern(ns), self.interner.intern(name));
        let handlers = &self.externs.handlers[&qref];
        let ExternHandler::Sync(handler) = &handlers[0] else {
            panic!("{ns}::{name} is not a sync handler")
        };
        handler(&self.rt, &mut args)
    }

    fn string(&self, s: &str) -> V {
        // SAFETY: stored as itself.
        unsafe { self.rt.erase::<String>(s.to_owned()) }
    }

    fn declared_return(&self, ns: &str, name: &str) -> PolyTy {
        let qref = QualifiedRef::qualified(self.interner.intern(ns), self.interner.intern(name));
        let func = self
            .externs
            .functions
            .iter()
            .find(|f| f.qref == qref)
            .unwrap_or_else(|| panic!("{ns}::{name} is declared"));
        let TyTerm::Fn { ret, .. } = &func.ty else {
            panic!("{ns}::{name} has a function type")
        };
        (**ret).clone()
    }
}

fn vec_of(elem: PolyTy, interner: &Interner) -> PolyTy {
    PolyTy::UserDefined {
        id: QualifiedRef::root(interner.intern("Vec")),
        type_args: vec![TypeArg::uniform(elem)],
        effect_args: vec![],
        identity_args: vec![],
    }
}

// -- The checked exit ---------------------------------------------------

#[test]
#[should_panic(expected = "expected a value erased from `alloc::string::String`")]
fn from_value_on_a_value_of_another_type_panics_naming_the_expected_type() {
    let rt = Counting::default();
    // SAFETY: stored as itself.
    let holds_an_i64 = unsafe { rt.erase::<i64>(7) };
    Erased::<Counting, String>::from_value(&rt, holds_an_i64);
}

#[test]
fn from_value_on_a_value_of_the_type_is_the_value() {
    let rt = Counting::default();
    // SAFETY: stored as itself.
    let holds_a_string = unsafe { rt.erase::<String>("s".to_owned()) };
    let erased = Erased::<Counting, String>::from_value(&rt, holds_a_string);
    assert_eq!(erased.as_ref(&rt), "s");
}

// -- E1 -----------------------------------------------------------------

#[test]
fn split_str_is_typed_vec_of_string() {
    let w = World::new();
    assert_eq!(
        w.declared_return("string", "split_str"),
        vec_of(PolyTy::String, &w.interner)
    );
}

#[test]
fn split_str_boxes_each_element_once_and_the_vec_once() {
    let w = World::new();
    let args = vec![w.string("a,b,c"), w.string(",")];
    let start = w.rt.counts();
    let _parts = w.call("string", "split_str", args);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 4,
            unboxes: 2
        }
    );
}

#[test]
fn reverse_on_the_vec_is_one_unbox_and_one_box() {
    let w = World::new();
    let parts = w.call(
        "string",
        "split_str",
        vec![w.string("a,b,c"), w.string(",")],
    );
    let start = w.rt.counts();
    let _reversed = w.call("vec", "reverse", vec![parts]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 1,
            unboxes: 1
        }
    );
}

#[test]
fn join_reads_through_as_ref_with_no_unbox() {
    let w = World::new();
    let parts = w.call(
        "string",
        "split_str",
        vec![w.string("a,b,c"), w.string(",")],
    );
    let reversed = w.call("vec", "reverse", vec![parts]);
    // SAFETY: `reversed` is live and unmoved for the call.
    let lent = unsafe { w.rt.reference(&reversed) };
    let start = w.rt.counts();
    let joined = w.call("t", "join_erased", vec![lent]);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 1,
            unboxes: 0
        }
    );
    // SAFETY: `join_erased` returns a String.
    assert_eq!(unsafe { w.rt.materialize::<String>(joined) }, "c,b,a");
}

impl Counting {
    fn no_closures(&self) -> V {
        panic!("Counting runs no closures")
    }
}
