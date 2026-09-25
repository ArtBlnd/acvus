//! `Erased<R, String>` at the boundary (E1): a `Vec` of it crosses as one
//! box, and a body reads its elements without taking any back. The
//! numbers are counted by `Counting`, a runtime whose `erase` and
//! `materialize` count every Rust value they box or unbox.

use acvus_extern::Ctx;
use std::any::{Any, TypeId, type_name};
use std::future::Ready;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_ext::{iterator_registry, string_registry, vec_registry};
use acvus_extern::{
    DirectOp, Erased, ExternHandler, Externs, Interner, OneValue, Owned, PolyTy, QualifiedRef,
    Registry, Release, Runtime, TyTerm, TypeArg, Words, extern_fn, extern_registry,
};

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
    Boxed(*mut (dyn Any + Send + Sync)),
    Reference(*const V),
    /// One register of the pair a slice or a `&str` view sits in.
    Word(u64),
}

// SAFETY: a cell is reached only through the value that owns it, and a
// `Reference` only while its target is live (RFC-0018).
unsafe impl Send for V {}
unsafe impl Sync for V {}

impl Release for V {
    fn release(self) {
        match self {
            V::Taken | V::None | V::Reference(_) | V::Word(_) => {}
            // SAFETY: `some` leaked this cell and nothing else releases it.
            V::Some(cell) => unsafe { *Box::from_raw(cell) }.release(),
            // SAFETY: `erase` leaked this cell and nothing else frees it.
            V::Boxed(cell) => drop(unsafe { Box::from_raw(cell) }),
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

// SAFETY: the test runtime's own value crosses as itself: `erase` and
// `materialize` hand the word through unchanged, and the capability is not
// used.
unsafe impl acvus_extern::OneValue<Counting> for V {
    fn erase(self, _: acvus_extern::Crossing<'_, Counting>) -> V {
        self
    }

    unsafe fn materialize(_: acvus_extern::Crossing<'_, Counting>, value: V) -> Self {
        value
    }
}

impl acvus_extern::Borrowable<Counting> for V {
    const LENDS_A_WORD: bool = false;

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


/// This test plays the runtime, which holds the crossing capability.
fn runtime_crossing(rt: &Counting) -> acvus_extern::Crossing<'_, Counting> {
    // SAFETY: the test is the runtime, and it crosses each value at the
    // type it was erased from.
    unsafe { acvus_extern::Crossing::new(rt) }
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
    /// This runtime's borrow is its storage's own Rust value, so a loan
    /// leaves nothing to re-encode.
    fn loan_ended(_: &mut V) {}

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
        open_ref(unsafe { <V as acvus_extern::Borrowable<Counting>>::deref(self, reference) })
    }

    unsafe fn deref_mut<'a, T>(&self, reference: &'a V) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        open_mut(unsafe { <V as acvus_extern::Borrowable<Counting>>::deref_mut(self, reference) })
    }

    unsafe fn encode(
        &self,
        _: &acvus_extern::Ty,
        _: &V,
        _: &mut Vec<u8>,
    ) -> acvus_extern::SpaceResult<()> {
        Err(acvus_extern::SpaceError::new("this test runtime lays out no value"))
    }

    unsafe fn reference(&self, target: &V) -> V {
        V::Reference(target as *const V)
    }
    fn rust_fn(&self, _: acvus_extern::RustBody<Self>) -> V {
        self.no_closures()
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

    unsafe fn result_at<'a>(&self, value: &'a V) -> Result<&'a V, &'a V> {
        let result = open_ref::<Result<acvus_extern::Owned<Self>, acvus_extern::Owned<Self>>>(value);
        result.as_ref().map(|ok| &**ok).map_err(|err| &**err)
    }

    unsafe fn result_at_mut<'a>(&self, value: &'a mut V) -> Result<&'a mut V, &'a mut V> {
        let result = open_mut::<Result<acvus_extern::Owned<Self>, acvus_extern::Owned<Self>>>(value);
        // SAFETY: the caller's contract carries `value_mut`'s.
        let payload = |held: &'a mut acvus_extern::Owned<Self>| unsafe {
            held.value_mut(acvus_extern::Holding::new())
        };
        result.as_mut().map(payload).map_err(payload)
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

    fn slice_into_run(&self, words: acvus_extern::Words, out: &mut [Self::Value]) {
        let [ptr, len] = words.into_pair();
        out[0] = V::Word(ptr);
        out[1] = V::Word(len);
    }

    unsafe fn slice_from_run(&self, run: &[Self::Value]) -> acvus_extern::Words {
        let (V::Word(ptr), V::Word(len)) = (run[0], run[1]) else {
            panic!("slice_from_run: not the pair a slice was written into: {run:?}")
        };
        // SAFETY: the caller's contract: `run` is what `slice_into_run` wrote
        // from `into_pair`.
        unsafe { acvus_extern::Words::from_pair([ptr, len]) }
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

// -- The reader under test --------------------------------------------

#[extern_fn(effect = pure)]
fn join_erased<Rt>(ctx: &mut Ctx<'_, Rt>, items: &[Erased<Rt, String>]) -> String
where
    Rt: Runtime,
{
    let rt = ctx.rt;
    let parts: Vec<&str> = items.iter().map(|s| s.as_ref(rt).as_str()).collect();
    parts.join(",")
}

/// The producer half of E1: a `Vec<Erased<Rt, String>>` built from `text`,
/// which is what the counts below are counts of.
#[extern_fn(effect = pure)]
fn split_erased<Rt>(ctx: &mut Ctx<'_, Rt>, text: &str, sep: &str) -> Vec<Erased<Rt, String>>
where
    Rt: Runtime,
{
    let rt = ctx.rt;
    text.split(sep)
        .map(|part| Erased::new(rt, part.to_owned()))
        .collect()
}

fn registries() -> Vec<Registry<Counting>> {
    vec![
        iterator_registry(),
        string_registry(),
        vec_registry(),
        extern_registry! {
            ns: "t",
            fns: [join_erased, split_erased],
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

    fn call(&self, ns: &str, name: &str, args: Vec<V>) -> V {
        let qref = QualifiedRef::qualified(self.interner.intern(ns), self.interner.intern(name));
        let handlers = &self.externs.handlers[&qref];
        let ExternHandler::Sync(handler) = &handlers[0] else {
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

    /// The pair a `&str` parameter is passed in (RFC-0062): the borrow is
    /// of `s`, which the caller keeps alive across the call.
    fn slice_view(&self, container: &V) -> [V; 2] {
        let elements = open_ref::<Vec<Owned<Counting>>>(container);
        let mut pair = [V::default(); 2];
        let words = Words::of_slice(elements);
        self.rt.slice_into_run(words, &mut pair);
        pair
    }

    fn str_view(&self, s: &str) -> [V; 2] {
        let mut pair = [V::default(); 2];
        let words = Words::of_str(s);
        self.rt.slice_into_run(words, &mut pair);
        pair
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
        region_params: 0,
    }
}

// -- The checked exit ---------------------------------------------------

#[test]
fn a_value_materialized_at_its_type_reads_back() {
    let rt = Counting::default();
    // SAFETY: stored as itself.
    let holds_a_string = unsafe { rt.erase::<String>("s".to_owned()) };
    // SAFETY: `holds_a_string` was just erased from a `String`.
    let erased = unsafe {
        <Erased<Counting, String> as OneValue<Counting>>::materialize(
            runtime_crossing(&rt),
            holds_a_string,
        )
    };
    assert_eq!(erased.as_ref(&rt), "s");
}

// -- E1 -----------------------------------------------------------------

#[test]
fn a_vec_of_erased_string_is_declared_as_a_vec_of_string() {
    let w = World::new();
    assert_eq!(
        w.declared_return("t", "split_erased"),
        vec_of(PolyTy::String, &w.interner)
    );
}

#[test]
fn a_returned_vec_boxes_each_element_once_and_the_vec_once() {
    let w = World::new();
    let (text, sep) = ("a,b,c".to_owned(), ",".to_owned());
    let args = [w.str_view(&text), w.str_view(&sep)].concat();
    let start = w.rt.counts();
    let _parts = w.call("t", "split_erased", args);
    assert_eq!(
        w.rt.since(start),
        Counts {
            boxes: 4,
            unboxes: 0
        }
    );
}

#[test]
fn reverse_on_the_vec_is_one_unbox_and_one_box() {
    let w = World::new();
    let (text, sep) = ("a,b,c".to_owned(), ",".to_owned());
    let parts = w.call(
        "t",
        "split_erased",
        [w.str_view(&text), w.str_view(&sep)].concat(),
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
    let (text, sep) = ("a,b,c".to_owned(), ",".to_owned());
    let parts = w.call(
        "t",
        "split_erased",
        [w.str_view(&text), w.str_view(&sep)].concat(),
    );
    let reversed = w.call("vec", "reverse", vec![parts]);
    let lent = w.slice_view(&reversed);
    let start = w.rt.counts();
    let joined = w.call("t", "join_erased", lent.to_vec());
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
