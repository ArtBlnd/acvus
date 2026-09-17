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
use acvus_extern::{CallToken, Erased, Interner, Runtime, Trap};

static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

#[derive(Default)]
struct Word(u64);

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

impl acvus_extern::Cross<Words> for Word {
    fn erase(self, _: &Words) -> Word {
        self
    }

    unsafe fn materialize(_: &Words, value: Word) -> Self {
        value
    }
}

impl acvus_extern::FromValue<Words> for Word {
    fn from_value(_: &Words, value: Word) -> Result<Word, Trap> {
        Ok(value)
    }
}

impl Runtime for Words {
    type Value = Word;
    type Error = Trap;
    type CallFuture<'a> = Ready<Result<Word, Trap>>;

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
        Word(unsafe { into_word(value) })
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

    fn call_is_sync(&self, _: &Word) -> bool {
        false
    }

    fn call_now(&self, _: &Word, _: &mut [Word], _: CallToken) -> Result<Word, Trap> {
        Err(Trap::internal("Words holds no closures"))
    }

    fn call_0<'a>(&'a self, _: &'a Word, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(Err(Trap::internal("Words holds no closures")))
    }

    fn call_1<'a>(&'a self, _: &'a Word, _: Word, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(Err(Trap::internal("Words holds no closures")))
    }

    fn call_n<'a>(&'a self, _: &'a Word, _: &mut [Word], _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(Err(Trap::internal("Words holds no closures")))
    }
}

macro_rules! define_tag {
    ($($name:ident: $t:ty),*) => {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Tag { $($name,)* }

        impl Tag {
            fn of<T: 'static>() -> Option<Tag> {
                let id = TypeId::of::<T>();
                $(if id == TypeId::of::<$t>() {
                    return Some(Tag::$name);
                })*
                None
            }

            fn erased_type(self) -> TypeId {
                match self { $(Tag::$name => TypeId::of::<$t>(),)* }
            }

            fn name(self) -> &'static str {
                match self { $(Tag::$name => stringify!($t),)* }
            }
        }
    };
}

acvus_extern::for_each_inline!(define_tag);

#[derive(Default)]
enum TaggedWord {
    #[default]
    Taken,
    Small(Tag, u64),
    Large(Box<Box<dyn Any + Send + Sync>>),
}

const _: () = assert!(size_of::<Word>() == 8);
const _: () = assert!(size_of::<TaggedWord>() == 16);

#[derive(Clone, Copy)]
struct Tags;

impl acvus_extern::Cross<Tags> for TaggedWord {
    fn erase(self, _: &Tags) -> TaggedWord {
        self
    }

    unsafe fn materialize(_: &Tags, value: TaggedWord) -> Self {
        value
    }
}

impl acvus_extern::FromValue<Tags> for TaggedWord {
    fn from_value(_: &Tags, value: TaggedWord) -> Result<TaggedWord, Trap> {
        Ok(value)
    }
}

/// # Safety
/// The value was erased from a `T`.
unsafe fn read<T>(value: &TaggedWord) -> &T
where
    T: 'static,
{
    match value {
        // SAFETY: the caller's contract.
        TaggedWord::Small(_, word) => unsafe { &*(&raw const *word).cast::<T>() },
        TaggedWord::Large(any) => any.downcast_ref::<T>().expect("erased from this type"),
        TaggedWord::Taken => panic!("read of a value that was taken"),
    }
}

/// # Safety
/// As `read`, exclusively.
unsafe fn read_mut<T>(value: &mut TaggedWord) -> &mut T
where
    T: 'static,
{
    match value {
        // SAFETY: the caller's contract.
        TaggedWord::Small(_, word) => unsafe { &mut *(&raw mut *word).cast::<T>() },
        TaggedWord::Large(any) => any.downcast_mut::<T>().expect("erased from this type"),
        TaggedWord::Taken => panic!("read of a value that was taken"),
    }
}

impl Runtime for Tags {
    type Value = TaggedWord;
    type Error = Trap;
    type CallFuture<'a> = Ready<Result<TaggedWord, Trap>>;

    fn type_of(&self, value: &TaggedWord) -> Option<TypeId> {
        match value {
            TaggedWord::Small(tag, _) => Some(tag.erased_type()),
            TaggedWord::Large(any) => Some((***any).type_id()),
            TaggedWord::Taken => None,
        }
    }

    fn type_name_of(&self, value: &TaggedWord) -> Option<&'static str> {
        match value {
            TaggedWord::Small(tag, _) => Some(tag.name()),
            TaggedWord::Large(_) | TaggedWord::Taken => None,
        }
    }

    unsafe fn materialize<T>(&self, value: TaggedWord) -> T
    where
        T: Send + Sync + 'static,
    {
        match (Tag::of::<T>(), value) {
            // SAFETY: the caller's contract.
            (Some(_), TaggedWord::Small(_, word)) => unsafe { out_of_word::<T>(word) },
            (None, TaggedWord::Large(any)) => {
                *(*any).downcast::<T>().expect("erased from this type")
            }
            _ => panic!(
                "materialize: {} is not how this value is held",
                std::any::type_name::<T>()
            ),
        }
    }

    unsafe fn erase<T>(&self, value: T) -> TaggedWord
    where
        T: Send + Sync + 'static,
    {
        match Tag::of::<T>() {
            // SAFETY: an `Inline` `T` fits the word and needs no drop.
            Some(tag) => TaggedWord::Small(tag, unsafe { into_word(value) }),
            None => TaggedWord::Large(Box::new(Box::new(value))),
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

    fn call_is_sync(&self, _: &TaggedWord) -> bool {
        false
    }

    fn call_now(
        &self,
        _: &TaggedWord,
        _: &mut [TaggedWord],
        _: CallToken,
    ) -> Result<TaggedWord, Trap> {
        Err(Trap::internal("Tags holds no closures"))
    }

    fn call_0<'a>(&'a self, _: &'a TaggedWord, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(Err(Trap::internal("Tags holds no closures")))
    }

    fn call_1<'a>(
        &'a self,
        _: &'a TaggedWord,
        _: TaggedWord,
        _: CallToken,
    ) -> Self::CallFuture<'a> {
        std::future::ready(Err(Trap::internal("Tags holds no closures")))
    }

    fn call_n<'a>(
        &'a self,
        _: &'a TaggedWord,
        _: &mut [TaggedWord],
        _: CallToken,
    ) -> Self::CallFuture<'a> {
        std::future::ready(Err(Trap::internal("Tags holds no closures")))
    }
}

async fn iter_range_sum<Rt>(rt: &Rt, n: i64) -> i64
where
    Rt: Runtime,
    Rt::Error: std::fmt::Debug,
{
    let mut next = 0i64;
    let mut it: Iter<Erased<Rt, i64>, (), (), Rt> = Iter::generate(move |rt| {
        (next < n).then(|| {
            let item = Erased::new(rt, black_box(next));
            next += 1;
            item
        })
    });

    let mut acc = 0i64;
    while let Some(item) = it.next(rt).await.expect("the pipeline does not trap") {
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
                rt.block_on(iter_range_sum(&Words, black_box(n)))
            }),
        ),
        (
            "iter<tagged>",
            timed(reps, expected, || {
                rt.block_on(iter_range_sum(&Tags, black_box(n)))
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
