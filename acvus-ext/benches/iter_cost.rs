//! Where the per-element cost of a lazy `Iter` pipeline sits: the dynamic
//! stage against the runtime value's tag (RFC-0044, stage 4).
//!
//! The same `range(0, n) | sum` pipeline runs over two runtimes that differ
//! in one thing, their value representation, and beside a plain Rust
//! iterator. `Tags` holds an inline value as a tag plus a word and anything
//! else in a box, which is the shape `acvus-interpreter`'s `Value` has;
//! change that shape there and this bench stops measuring it.

use std::any::{Any, TypeId};
use std::future::Ready;
use std::hint::black_box;
use std::mem::ManuallyDrop;
use std::time::{Duration, Instant};

use acvus_ext::Iter;
use acvus_extern::Ctx;
use acvus_extern::{Erased, Interner, Runtime};

/// No registry these tests combine declares a sliceable container, so the
/// pair a slice would occupy is never built or read.
const NO_SLICES: &str = "this runtime holds no slices";

static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

/// `Word` stays eight bytes, so absence has to live in a niche rather than
/// in a tag beside it: the stage-4 microbench measured a 24-byte value at
/// 22.5 ns/element against 7.1 for a 16-byte one, and widening this one
/// would measure the widening.
#[derive(Clone, Copy)]
struct Word(u64);

/// A word owns nothing, so leaving a register releases nothing (RFC-0048
/// §4): what makes this runtime the floor the tagged one is measured against.
impl acvus_extern::Release for Word {
    fn release(self) {}
}

/// Not built: these runtimes carry no language `Option`.
fn no_options() -> ! {
    panic!("the iter_cost runtimes hold no language Option")
}

/// Not built: these runtimes carry no language enum, so no value of theirs is a
/// variant and no tag register is written or read.
fn no_variants() -> ! {
    panic!("the iter_cost runtimes hold no variants")
}

impl Default for Word {
    fn default() -> Self {
        Word(u64::MAX)
    }
}

#[derive(Clone, Copy)]
struct Words;

/// # Safety
/// `T` is one of the `Inline` types, so it fits the word and needs no drop.
unsafe fn into_word<T>(value: T) -> u64
where
    T: Send + Sync + 'static,
{
    let value = ManuallyDrop::new(value);
    // SAFETY: the caller's contract.
    unsafe { *(&raw const *value).cast::<u64>() }
}

/// # Safety
/// The word was written by `into_word::<T>`.
unsafe fn out_of_word<T>(word: u64) -> T
where
    T: Send + Sync + 'static,
{
    // SAFETY: the caller's contract.
    unsafe { (&raw const word).cast::<T>().read() }
}

acvus_extern::cross_one_value!(Word, at Words);

impl acvus_extern::OneValue<Words> for Word {
    fn erase(self, _: &Words) -> Word {
        self
    }

    unsafe fn materialize(_: &Words, value: Word) -> Self {
        value
    }
}

// SAFETY: `Word` is this runtime's own value, which no Rust type disagrees with.
unsafe impl acvus_extern::FromValue<Words> for Word {
    unsafe fn from_value(_: &Words, value: Word) -> Word {
        value
    }
}

impl Runtime for Words {
    fn instance_value(_: acvus_extern::InstanceRun) -> Self::Value {
        panic!("Words declares no instances")
    }

    unsafe fn instance_run(_: &Self::Value) -> acvus_extern::InstanceRun {
        panic!("Words declares no instances")
    }

    type Op = acvus_extern::DirectOp<Words>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = acvus_extern::DirectOp<Words>;
    type FusedShape = ();

    acvus_extern::direct_call_forms!();

    type Value = Word;
    type Frame<'a> = ();
    type Rooted<'a> = acvus_extern::Ctx<'a, Self>;
    type CallFuture<'a> = Ready<Word>;

    fn rooted(&self) -> acvus_extern::Ctx<'_, Self> {
        acvus_extern::Ctx::new(self, ())
    }
    fn ctx_of<'a, 'r>(
        rooted: &'r mut acvus_extern::Ctx<'a, Self>,
    ) -> &'r mut acvus_extern::Ctx<'a, Self>
    where
        'a: 'r,
    {
        rooted
    }

    fn type_of(&self, _: &Word) -> Option<TypeId> {
        Some(TypeId::of::<i64>())
    }

    fn type_name_of(&self, _: &Word) -> Option<&'static str> {
        Some("i64")
    }

    unsafe fn materialize<T>(&self, value: Word) -> T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        unsafe { out_of_word::<T>(value.0) }
    }

    unsafe fn erase<T>(&self, value: T) -> Word
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        let word = unsafe { into_word(value) };
        debug_assert_ne!(word, u64::MAX, "the sentinel word is not an erased value");
        Word(word)
    }

    unsafe fn value_as_ref<'a, T>(&'a self, value: &'a Word) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        unsafe { &*(&raw const value.0).cast::<T>() }
    }

    unsafe fn value_as_mut<'a, T>(&'a self, value: &'a mut Word) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        unsafe { &mut *(&raw mut value.0).cast::<T>() }
    }

    unsafe fn inline_ref<T>(value: &Word) -> &T
    where
        T: acvus_extern::Inline,
    {
        // SAFETY: the caller's contract.
        unsafe { &*(&raw const value.0).cast::<T>() }
    }

    unsafe fn inline_mut<T>(value: &mut Word) -> &mut T
    where
        T: acvus_extern::Inline,
    {
        // SAFETY: the caller's contract.
        unsafe { &mut *(&raw mut value.0).cast::<T>() }
    }

    unsafe fn deref<'a, T>(&self, _: &'a Word) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        panic!("Words holds no references")
    }

    unsafe fn deref_mut<'a, T>(&self, _: &'a Word) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        panic!("Words holds no references")
    }

    unsafe fn reference(&self, _: &Word) -> Word {
        panic!("Words holds no references")
    }

    fn symbol(&self, name: &str) -> acvus_utils::Astr {
        SYMBOLS.intern(name)
    }

    fn variant_tag(&self, _: &str) -> Self::Value {
        no_variants()
    }

    unsafe fn tag_symbol(&self, _: &Self::Value) -> acvus_utils::Astr {
        no_variants()
    }

    fn undef(&self) -> Self::Value {
        no_variants()
    }

    fn is_undef(&self, _: &Self::Value) -> bool {
        no_variants()
    }

    fn slice_into_run(&self, _: acvus_extern::Words, _: &mut [Self::Value]) {
        panic!("{NO_SLICES}")
    }

    unsafe fn slice_from_run(&self, _: &[Self::Value]) -> acvus_extern::Words {
        panic!("{NO_SLICES}")
    }

    fn none(&self) -> Word {
        no_options()
    }
    fn some(&self, _: Word) -> Word {
        no_options()
    }
    fn is_none(&self, _: &Word) -> bool {
        no_options()
    }
    fn unwrap_some(&self, _: Word) -> Word {
        no_options()
    }
    unsafe fn some_at<'a>(&self, _: &'a Word) -> Option<&'a Word> {
        no_options()
    }
    unsafe fn some_at_mut<'a>(&self, _: &'a mut Word) -> Option<&'a mut Word> {
        no_options()
    }
    fn call_is_sync(&self, _: &Word) -> bool {
        false
    }

    unsafe fn call_now<A>(&self, _: &Word, _: &mut Ctx<'_, Words>, _: A) -> Word
    where
        A: acvus_extern::IntoRun<Self>,
    {
        self.no_closures()
    }

    unsafe fn call_0<'a>(&'a self, _: &'a Word) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }

    unsafe fn call_1<'a>(&'a self, _: &'a Word, _: Word) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }

    unsafe fn call_n<'a>(&'a self, _: &'a Word, _: &mut [Word]) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }
}

impl Words {
    fn no_closures(&self) -> Word {
        panic!("Words holds no closures")
    }
}

macro_rules! define_kind {
    ($($name:ident: $t:ty),*) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum Kind { Taken, Large, $($name,)* }

        impl Kind {
            fn of<T: 'static>() -> Option<Kind> {
                let id = TypeId::of::<T>();
                $(if id == TypeId::of::<$t>() {
                    return Some(Kind::$name);
                })*
                None
            }

            fn erased_type(self) -> Option<TypeId> {
                match self {
                    $(Kind::$name => Some(TypeId::of::<$t>()),)*
                    Kind::Taken | Kind::Large => None,
                }
            }

            fn name(self) -> Option<&'static str> {
                match self {
                    $(Kind::$name => Some(stringify!($t)),)*
                    Kind::Taken | Kind::Large => None,
                }
            }

            fn is_inline(self) -> bool {
                match self {
                    $(Kind::$name)|* => true,
                    Kind::Taken | Kind::Large => false,
                }
            }
        }
    };
}

acvus_extern::for_each_inline!(define_kind);

type Payload = Box<dyn Any + Send + Sync>;

/// The same shape as the interpreter's `Value`: one kind byte and one word,
/// so this bench prices the tag the interpreter actually pays.
#[derive(Clone, Copy)]
#[repr(C)]
struct TaggedWord {
    kind: Kind,
    word: u64,
}

const _: () = assert!(size_of::<Word>() == 8);
const _: () = assert!(size_of::<TaggedWord>() == 16);

impl Default for TaggedWord {
    fn default() -> Self {
        TaggedWord {
            kind: Kind::Taken,
            word: 0,
        }
    }
}

impl TaggedWord {
    /// # Safety
    /// The word was written by `erase` for a type with no inline kind.
    unsafe fn payload(&self) -> &Payload {
        debug_assert_eq!(self.kind, Kind::Large, "payload: not a large value");
        // SAFETY: the caller's contract; the word is the pointer `erase` leaked.
        unsafe { &*(self.word as *const Payload) }
    }
}

/// RFC-0048 §4: a runtime value is `Copy` and owns nothing by falling out
/// of scope. The `Owned<R>` at every Rust store is what calls this.
impl acvus_extern::Release for TaggedWord {
    fn release(self) {
        if self.kind == Kind::Large {
            // SAFETY: the word is the pointer `erase` leaked, released once.
            drop(unsafe { Box::from_raw(self.word as *mut Payload) });
        }
    }
}

#[derive(Clone, Copy)]
struct Tags;

acvus_extern::cross_one_value!(TaggedWord, at Tags);

impl acvus_extern::OneValue<Tags> for TaggedWord {
    fn erase(self, _: &Tags) -> TaggedWord {
        self
    }

    unsafe fn materialize(_: &Tags, value: TaggedWord) -> Self {
        value
    }
}

// SAFETY: `TaggedWord` is this runtime's own value, which no Rust type disagrees with.
unsafe impl acvus_extern::FromValue<Tags> for TaggedWord {
    unsafe fn from_value(_: &Tags, value: TaggedWord) -> TaggedWord {
        value
    }
}

/// # Safety
/// The value was erased from a `T`.
unsafe fn read<T>(value: &TaggedWord) -> &T
where
    T: 'static,
{
    match value.kind {
        // SAFETY: the caller's contract.
        Kind::Large => unsafe { value.payload() }
            .downcast_ref::<T>()
            .expect("erased from this type"),
        Kind::Taken => panic!("read of a value that was taken"),
        // SAFETY: the caller's contract.
        _ => unsafe { &*(&raw const value.word).cast::<T>() },
    }
}

/// # Safety
/// As `read`, exclusively.
unsafe fn read_mut<T>(value: &mut TaggedWord) -> &mut T
where
    T: 'static,
{
    match value.kind {
        // SAFETY: the word is the pointer `erase` leaked and `value` is held exclusively.
        Kind::Large => unsafe { &mut *(value.word as *mut Payload) }
            .downcast_mut::<T>()
            .expect("erased from this type"),
        Kind::Taken => panic!("read of a value that was taken"),
        // SAFETY: the caller's contract.
        _ => unsafe { &mut *(&raw mut value.word).cast::<T>() },
    }
}

impl Runtime for Tags {
    fn instance_value(_: acvus_extern::InstanceRun) -> Self::Value {
        panic!("Tags declares no instances")
    }

    unsafe fn instance_run(_: &Self::Value) -> acvus_extern::InstanceRun {
        panic!("Tags declares no instances")
    }

    type Op = acvus_extern::DirectOp<Tags>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = acvus_extern::DirectOp<Tags>;
    type FusedShape = ();

    acvus_extern::direct_call_forms!();

    type Value = TaggedWord;
    type Frame<'a> = ();
    type Rooted<'a> = acvus_extern::Ctx<'a, Self>;
    type CallFuture<'a> = Ready<TaggedWord>;

    fn rooted(&self) -> acvus_extern::Ctx<'_, Self> {
        acvus_extern::Ctx::new(self, ())
    }
    fn ctx_of<'a, 'r>(
        rooted: &'r mut acvus_extern::Ctx<'a, Self>,
    ) -> &'r mut acvus_extern::Ctx<'a, Self>
    where
        'a: 'r,
    {
        rooted
    }

    fn type_of(&self, value: &TaggedWord) -> Option<TypeId> {
        match value.kind {
            // SAFETY: the word is the pointer `erase` leaked.
            Kind::Large => Some((**unsafe { value.payload() }).type_id()),
            kind => kind.erased_type(),
        }
    }

    fn type_name_of(&self, value: &TaggedWord) -> Option<&'static str> {
        value.kind.name()
    }

    unsafe fn materialize<T>(&self, value: TaggedWord) -> T
    where
        T: Send + Sync + 'static,
    {
        let value = ManuallyDrop::new(value);
        match Kind::of::<T>() {
            Some(kind) => {
                debug_assert_eq!(value.kind, kind, "materialize: not how this value is held");
                // SAFETY: the caller's contract.
                unsafe { out_of_word::<T>(value.word) }
            }
            None => {
                debug_assert_eq!(
                    value.kind,
                    Kind::Large,
                    "materialize: not how this value is held"
                );
                // SAFETY: the word is the pointer `erase` leaked, taken once.
                let payload = unsafe { Box::from_raw(value.word as *mut Payload) };
                *(*payload).downcast::<T>().expect("erased from this type")
            }
        }
    }

    unsafe fn erase<T>(&self, value: T) -> TaggedWord
    where
        T: Send + Sync + 'static,
    {
        match Kind::of::<T>() {
            Some(kind) => TaggedWord {
                kind,
                // SAFETY: an `Inline` `T` fits the word and needs no drop.
                word: unsafe { into_word(value) },
            },
            None => TaggedWord {
                kind: Kind::Large,
                word: Box::into_raw(Box::new(Box::new(value) as Payload)) as u64,
            },
        }
    }

    unsafe fn value_as_ref<'a, T>(&'a self, value: &'a TaggedWord) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        unsafe { read::<T>(value) }
    }

    unsafe fn value_as_mut<'a, T>(&'a self, value: &'a mut TaggedWord) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract.
        unsafe { read_mut::<T>(value) }
    }

    unsafe fn inline_ref<T>(value: &TaggedWord) -> &T
    where
        T: acvus_extern::Inline,
    {
        // SAFETY: the caller's contract.
        unsafe { read::<T>(value) }
    }

    unsafe fn inline_mut<T>(value: &mut TaggedWord) -> &mut T
    where
        T: acvus_extern::Inline,
    {
        // SAFETY: the caller's contract.
        unsafe { read_mut::<T>(value) }
    }

    unsafe fn deref<'a, T>(&self, _: &'a TaggedWord) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        panic!("Tags holds no references")
    }

    unsafe fn deref_mut<'a, T>(&self, _: &'a TaggedWord) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        panic!("Tags holds no references")
    }

    unsafe fn reference(&self, _: &TaggedWord) -> TaggedWord {
        panic!("Tags holds no references")
    }

    fn symbol(&self, name: &str) -> acvus_utils::Astr {
        SYMBOLS.intern(name)
    }

    fn variant_tag(&self, _: &str) -> Self::Value {
        no_variants()
    }

    unsafe fn tag_symbol(&self, _: &Self::Value) -> acvus_utils::Astr {
        no_variants()
    }

    fn undef(&self) -> Self::Value {
        no_variants()
    }

    fn is_undef(&self, _: &Self::Value) -> bool {
        no_variants()
    }

    fn slice_into_run(&self, _: acvus_extern::Words, _: &mut [Self::Value]) {
        panic!("{NO_SLICES}")
    }

    unsafe fn slice_from_run(&self, _: &[Self::Value]) -> acvus_extern::Words {
        panic!("{NO_SLICES}")
    }

    fn none(&self) -> TaggedWord {
        no_options()
    }
    fn some(&self, _: TaggedWord) -> TaggedWord {
        no_options()
    }
    fn is_none(&self, _: &TaggedWord) -> bool {
        no_options()
    }
    fn unwrap_some(&self, _: TaggedWord) -> TaggedWord {
        no_options()
    }
    unsafe fn some_at<'a>(&self, _: &'a TaggedWord) -> Option<&'a TaggedWord> {
        no_options()
    }
    unsafe fn some_at_mut<'a>(&self, _: &'a mut TaggedWord) -> Option<&'a mut TaggedWord> {
        no_options()
    }
    fn call_is_sync(&self, _: &TaggedWord) -> bool {
        false
    }

    unsafe fn call_now<A>(&self, _: &TaggedWord, _: &mut Ctx<'_, Tags>, _: A) -> TaggedWord
    where
        A: acvus_extern::IntoRun<Self>,
    {
        self.no_closures()
    }

    unsafe fn call_0<'a>(&'a self, _: &'a TaggedWord) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }

    unsafe fn call_1<'a>(&'a self, _: &'a TaggedWord, _: TaggedWord) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }

    unsafe fn call_n<'a>(
        &'a self,
        _: &'a TaggedWord,
        _: &mut [TaggedWord],
    ) -> Self::CallFuture<'a> {
        std::future::ready(self.no_closures())
    }
}

impl Tags {
    fn no_closures(&self) -> TaggedWord {
        panic!("Tags holds no closures")
    }
}

async fn iter_range_sum<Rt>(ctx: &mut Ctx<'_, Rt>, n: i64) -> i64
where
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut next = 0i64;
    let mut it: Iter<Erased<Rt, i64>, (), (), Rt> = Iter::generate(move |rt| {
        (next < n).then(|| {
            let item = Erased::new(rt, black_box(next));
            next += 1;
            item
        })
    });

    let mut acc = 0i64;
    while let Some(item) = it.next(ctx).await {
        acc += *item.as_ref(rt);
    }
    acc
}

fn rust_range_sum(n: i64) -> i64 {
    (0..n).map(black_box).sum()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn timed<F>(reps: usize, expected: i64, mut run: F) -> Duration
where
    F: FnMut() -> i64,
{
    let mut samples = Vec::new();
    for rep in 0..reps {
        let start = Instant::now();
        let value = black_box(run());
        let elapsed = start.elapsed();
        assert_eq!(value, expected, "the pipeline summed to {value}");
        if rep > 0 {
            samples.push(elapsed);
        }
    }
    median(samples)
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");
    let n: i64 = 1_000_000;
    let reps = 11;
    let expected = n * (n - 1) / 2;

    let rows = [
        (
            "rust",
            timed(reps, expected, || rust_range_sum(black_box(n))),
        ),
        (
            "iter<word>",
            timed(reps, expected, || {
                rt.block_on(iter_range_sum(&mut Ctx::new(&Words, ()), black_box(n)))
            }),
        ),
        (
            "iter<tagged>",
            timed(reps, expected, || {
                rt.block_on(iter_range_sum(&mut Ctx::new(&Tags, ()), black_box(n)))
            }),
        ),
    ];

    println!("{:>14} {:>14} {:>16}", "case", "total/us", "ns/element");
    for (name, elapsed) in rows {
        println!(
            "{:>14} {:>14.1} {:>16.2}",
            name,
            elapsed.as_secs_f64() * 1e6,
            elapsed.as_secs_f64() * 1e9 / n as f64
        );
    }
}
