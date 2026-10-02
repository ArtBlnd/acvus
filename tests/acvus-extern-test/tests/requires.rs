//! A requirement's value (rfc_extern impls §3): the impl a call's site
//! chose, held as a handle at the type where its generic function takes its
//! receiver by `&`, and with its receiver where it takes it by `&mut`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::callback::CallsBack;
use acvus_extern::decl::{DeclTy, EffectVar, Interner, Requirement, TypeVar};
use acvus_extern::done::Done;
use acvus_extern::executor::Executor;
use acvus_extern::instance::{Instance, InstanceOf};
use acvus_extern::register::{Declared, ExternFn, GenericDeclared};
use acvus_extern::runtime::{Place, Runtime};
use acvus_extern::{Entry, SiteShape, Uniform, extern_fn, generic_fn};
use acvus_extern_test::{Frame, Pass, Tally, WordRuntime};

#[generic_fn]
fn same<R: Runtime, T: TypeVar<R>>(a: &T, b: &T) -> bool;

#[extern_fn(effect = pure, impl_of = same<Uniform<R, i64>>)]
fn same_i64<R: Runtime>(#[executor] cx: &Executor<'_, R>, a: &Uniform<R, i64>, b: &Uniform<R, i64>) -> bool {
    a.get(cx) == b.get(cx)
}

#[extern_fn(effect = pure)]
fn contains<R: CallsBack, T: TypeVar<R>>(
    #[executor] cx: &Executor<'_, R>,
    #[requires] eq: &InstanceOf<R, same, (T,)>,
    xs: &[T],
    x: &T,
) -> bool {
    xs.iter().any(|y| eq.call(cx, (y, x)))
}

#[generic_fn]
fn bump<R: Runtime, T: TypeVar<R>>(counter: &mut T) -> u64;

#[extern_fn(impl_of = bump<Uniform<R, i64>>)]
fn bump_i64<R: Runtime>(#[executor] cx: &Executor<'_, R>, counter: &mut Uniform<R, i64>) -> u64 {
    let next = counter.get(cx) + 1;
    counter.set(cx, next);
    next as u64
}

#[extern_fn]
fn bump_twice<R: CallsBack, T: TypeVar<R>>(
    #[executor] cx: &Executor<'_, R>,
    #[instance] counter: Instance<R, bump, (T,)>,
) -> u64 {
    let mut counter = counter;
    counter.call(cx, ());
    counter.call(cx, ())
}

#[generic_fn]
fn spend<R: Runtime, T: TypeVar<R>>(coin: T) -> i64;

#[extern_fn(impl_of = spend<Uniform<R, i64>>)]
fn spend_i64<R: Runtime>(#[executor] cx: &Executor<'_, R>, coin: Uniform<R, i64>) -> i64 {
    coin.get(cx) * 10
}

#[extern_fn]
fn spend_now<R: CallsBack, T: TypeVar<R>>(#[executor] cx: &Executor<'_, R>, #[instance] coin: Instance<R, spend, (T,)>) -> i64 {
    coin.call_once(cx, ())
}

#[extern_fn]
async fn spend_later<R: CallsBack, T: TypeVar<R>>(
    #[executor] cx: &Executor<'_, R>,
    #[instance] coin: Instance<R, spend, (T,)>,
) -> i64 {
    coin.call_once_await(cx, ()).await
}

/// A variable holding `value`, made as a program makes one: a call's result.
fn produce<T: for<'l> Place<'l, WordRuntime>>(frame: &mut Frame<'_>, value: T) -> usize {
    frame.call([], |entry| entry.open(|cx, [], []| cx.finish(value)))
}

/// A variable's value, given to a call that takes it by value.
fn read<T: for<'l> Place<'l, WordRuntime>>(frame: &mut Frame<'_>, variable: usize) -> T {
    let mut read = None;
    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [given], []| {
            read = Some(given.open::<T>());
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

#[test]
fn a_handle_at_the_type_is_its_requirement() {
    let names = Interner::new();
    let contains = <contains as Declared<WordRuntime>>::declaration(&contains::new(), &names);
    let same = <same as GenericDeclared<WordRuntime>>::path(&names);

    assert_eq!(contains.params.len(), 2);
    assert_eq!(contains.requires, [Requirement { function: same, pattern: vec![DeclTy::Param(0)], fact: None }]);
}

#[test]
fn an_instance_is_an_argument_and_a_requirement() {
    let names = Interner::new();
    let bump_twice = <bump_twice as Declared<WordRuntime>>::declaration(&bump_twice::new(), &names);
    let bump = <bump as GenericDeclared<WordRuntime>>::path(&names);

    assert_eq!(bump_twice.params, [(names.intern("counter"), DeclTy::Param(0))]);
    assert_eq!(bump_twice.requires, [Requirement { function: bump, pattern: vec![DeclTy::Param(0)], fact: None }]);
}

#[test]
fn a_handle_calls_the_impl_its_site_chose_on_any_value() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let same = frame.function(same_i64::new());
    let xs = frame.array(&[3i64, 5, 8]);
    let five = produce(&mut frame, 5i64);
    let six = produce(&mut frame, 6i64);

    let found = frame.call_dispatch_choosing(&contains::new(), [Pass::Lend(xs), Pass::Lend(five)], [same]);
    let missed = frame.call_dispatch_choosing(&contains::new(), [Pass::Lend(xs), Pass::Lend(six)], [same]);

    assert!(read::<bool>(&mut frame, found));
    assert!(!read::<bool>(&mut frame, missed));
}

#[test]
fn an_instance_lends_its_own_receiver_to_each_call() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let bump = frame.function(bump_i64::new());
    let counter = produce(&mut frame, 40i64);

    let second = frame.call_dispatch_choosing(&bump_twice::new(), [Pass::Give(counter)], [bump]);

    assert_eq!(read::<u64>(&mut frame, second), 42);
}

#[test]
fn an_instance_gives_its_own_receiver_to_its_one_call() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let spend = frame.function(spend_i64::new());
    let now = produce(&mut frame, 4i64);
    let later = produce(&mut frame, 7i64);

    let spent = frame.call_dispatch_choosing(&spend_now::new(), [Pass::Give(now)], [spend]);
    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");
    let awaited = executor.block_on(frame.call_dispatch_awaited_choosing(&spend_later::new(), [Pass::Give(later)], [spend]));

    assert_eq!(read::<i64>(&mut frame, spent), 40);
    assert_eq!(read::<i64>(&mut frame, awaited), 70);
}

/// A value keeping an instance for later calls, as an iterator adapter keeps
/// the iterator it reads with its `next`.
#[derive(acvus_extern::Opaque)]
pub struct Bumper<R: Runtime, T: TypeVar<R>> {
    counter: Instance<R, bump, (T,)>,
    calls: u64,
}

#[extern_fn]
fn bumper<R: CallsBack, T: TypeVar<R>>(#[instance] counter: Instance<R, bump, (T,)>) -> Bumper<R, T> {
    Bumper { counter, calls: 0 }
}

#[extern_fn]
fn step<R: CallsBack, T: TypeVar<R>>(#[executor] cx: &Executor<'_, R>, b: &mut Bumper<R, T>) -> u64 {
    b.calls += 1;
    b.counter.call(cx, ())
}

#[test]
fn a_kept_instance_calls_its_impl_in_a_later_call() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let bump = frame.function(bump_i64::new());
    let counter = produce(&mut frame, 10i64);

    let kept = frame.call_dispatch_choosing(&bumper::new(), [Pass::Give(counter)], [bump]);
    frame.release(bump);
    let first = frame.call_dispatch(&step::new(), [Pass::Lend(kept)]);
    let second = frame.call_dispatch(&step::new(), [Pass::Lend(kept)]);

    assert_eq!(read::<u64>(&mut frame, first), 11);
    assert_eq!(read::<u64>(&mut frame, second), 12);
}

#[generic_fn]
fn tenfold<R: Runtime>(n: &u64) -> u64;

#[extern_fn(effect = pure, impl_of = tenfold)]
fn tenfold_u64(n: &u64) -> u64 {
    n * 10
}

#[extern_fn(effect = pure)]
fn tenfold_twice<R: CallsBack>(#[executor] cx: &Executor<'_, R>, #[requires] ten: &InstanceOf<R, tenfold, ()>, n: &u64) -> u64 {
    let once = ten.call(cx, (n,));
    ten.call(cx, (&once,))
}

#[test]
fn a_generic_function_of_no_variable_is_required_and_called() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let ten = frame.function(tenfold_u64::new());
    let three = produce(&mut frame, 3u64);

    let called = frame.call_dispatch_choosing(&tenfold_twice::new(), [Pass::Lend(three)], [ten]);

    assert_eq!(read::<u64>(&mut frame, called), 300);
}

#[generic_fn]
fn next<R: Runtime, I: TypeVar<R>, T: TypeVar<R>, E: EffectVar>(it: &mut I) -> Option<T>;

/// An iterator counting down from `remaining`: it yields `remaining`, then
/// one less, down to 1, and then nothing. The `Tally` counts its release.
#[derive(acvus_extern::Opaque)]
pub struct Countdown {
    remaining: i64,
    _released: Tally,
}

#[extern_fn(impl_of = next<Uniform<R, Countdown>, Uniform<R, i64>, E>)]
fn next_countdown<R: Runtime, E: EffectVar>(
    #[executor] cx: &Executor<'_, R>,
    it: &mut Uniform<R, Countdown>,
) -> Option<Uniform<R, i64>> {
    count_down(cx, it)
}

/// The countdown's step: what it yields next, and one less left.
fn count_down<R: Runtime>(cx: &Executor<'_, R>, it: &mut Uniform<R, Countdown>) -> Option<Uniform<R, i64>> {
    let countdown = it.open_mut(cx);
    if countdown.remaining == 0 {
        return None;
    }
    let yielded = countdown.remaining;
    countdown.remaining -= 1;
    Some(Uniform::new(cx, yielded))
}

#[extern_fn]
async fn collect<R: CallsBack, I: TypeVar<R>, T: TypeVar<R>, E: EffectVar>(
    #[executor] cx: &Executor<'_, R>,
    #[instance] it: Instance<R, next, (I, T, E)>,
) -> Vec<T> {
    let mut it = it;
    let mut items = Vec::new();
    while let Some(x) = it.call_await(cx, ()).await {
        items.push(x);
    }
    items
}

/// A variable's value, a sequence of held `i64`s, given to a call that reads
/// each.
fn read_held_i64s(frame: &mut Frame<'_>, variable: usize) -> Vec<i64> {
    let mut read = None;
    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [given], []| {
            let held = given.open::<Vec<Uniform<WordRuntime, i64>>>();
            read = Some(held.iter().map(|x| x.get(&cx)).collect());
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// A countdown with a tally of its releases.
fn countdown(remaining: i64) -> (Countdown, Arc<AtomicUsize>) {
    let releases = Arc::new(AtomicUsize::new(0));
    (Countdown { remaining, _released: Tally(Arc::clone(&releases)) }, releases)
}

#[test]
fn an_instance_with_an_effect_variable_is_called_awaited_in_a_loop() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let next = frame.function(next_countdown::new());
    let (three, releases) = countdown(3);
    let it = produce(&mut frame, three);

    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");
    let collected = executor.block_on(frame.call_dispatch_awaited_choosing(&collect::new(), [Pass::Give(it)], [next]));

    assert_eq!(read_held_i64s(&mut frame, collected), [3, 2, 1]);
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

/// A value keeping an iterator's instance, stepping it once per call.
#[derive(acvus_extern::Opaque)]
pub struct Walker<R: Runtime, I: TypeVar<R>, T: TypeVar<R>, E: EffectVar> {
    it: Instance<R, next, (I, T, E)>,
}

#[extern_fn]
fn walker<R: CallsBack, I: TypeVar<R>, T: TypeVar<R>, E: EffectVar>(
    #[instance] it: Instance<R, next, (I, T, E)>,
) -> Walker<R, I, T, E> {
    Walker { it }
}

#[extern_fn]
fn walk<R: CallsBack, I: TypeVar<R>, T: TypeVar<R>, E: EffectVar>(
    #[executor] cx: &Executor<'_, R>,
    w: &mut Walker<R, I, T, E>,
) -> Option<T> {
    w.it.call(cx, ())
}

#[extern_fn]
async fn walk_awaited<R: CallsBack, I: TypeVar<R>, T: TypeVar<R>, E: EffectVar>(
    #[executor] cx: &Executor<'_, R>,
    w: &mut Walker<R, I, T, E>,
) -> Option<T> {
    w.it.call_await(cx, ()).await
}

#[test]
fn an_instance_with_an_effect_variable_is_kept_in_an_opaque_value() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let next = frame.function(next_countdown::new());
    let (three, releases) = countdown(3);
    let it = produce(&mut frame, three);

    let kept = frame.call_dispatch_choosing(&walker::new(), [Pass::Give(it)], [next]);
    frame.release(next);
    let first = frame.call_dispatch(&walk::new(), [Pass::Lend(kept)]);
    let executor = tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor");
    let second = executor.block_on(frame.call_dispatch_awaited(&walk_awaited::new(), [Pass::Lend(kept)]));
    let third = frame.call_dispatch(&walk::new(), [Pass::Lend(kept)]);
    let past_the_end = frame.call_dispatch(&walk::new(), [Pass::Lend(kept)]);

    assert_eq!(read::<Option<i64>>(&mut frame, first), Some(3));
    assert_eq!(read::<Option<i64>>(&mut frame, second), Some(2));
    assert_eq!(read::<Option<i64>>(&mut frame, third), Some(1));
    assert_eq!(read::<Option<i64>>(&mut frame, past_the_end), None);
    assert_eq!(releases.load(Ordering::SeqCst), 0);

    frame.release(kept);

    assert_eq!(releases.load(Ordering::SeqCst), 1);
    drop(frame);

    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

/// `next_countdown`'s dispatch written by hand, as `#[extern_fn]` writes one,
/// as a function value that also holds a `Tally` counting its own release.
struct CountedNext {
    _released: Tally,
}

// SAFETY: this stands in for `#[extern_fn]` on `next_countdown`: `call`
// opens its one argument at the type `next_countdown` takes it,
// `&mut Uniform<R, Countdown>`, and gives the result `next_countdown` gives.
#[allow(unsafe_code)]
unsafe impl acvus_extern::seal::ExternFns for CountedNext {}

impl ExternFn<WordRuntime, 1, 0> for CountedNext {
    fn call(&self, runtime: &WordRuntime, args: [<WordRuntime as Runtime>::Value; 1], site: &SiteShape<WordRuntime, 0>) -> Done<WordRuntime> {
        Entry::new(runtime, args, site).open(|cx, [it], []| {
            let result = count_down(&cx, it.open::<&mut Uniform<WordRuntime, Countdown>>());
            cx.finish(result)
        })
    }
}

#[test]
fn an_instance_kept_in_an_opaque_value_releases_its_impl_once() {
    let runtime = WordRuntime::default();
    let impl_releases = Arc::new(AtomicUsize::new(0));
    let mut frame = Frame::new(&runtime);
    let next = frame.function(CountedNext { _released: Tally(Arc::clone(&impl_releases)) });
    let (two, receiver_releases) = countdown(2);
    let it = produce(&mut frame, two);

    let kept = frame.call_dispatch_choosing(&walker::new(), [Pass::Give(it)], [next]);
    frame.release(next);
    let first = frame.call_dispatch(&walk::new(), [Pass::Lend(kept)]);

    assert_eq!(read::<Option<i64>>(&mut frame, first), Some(2));
    assert_eq!(impl_releases.load(Ordering::SeqCst), 0);
    assert_eq!(receiver_releases.load(Ordering::SeqCst), 0);

    frame.release(kept);

    assert_eq!(impl_releases.load(Ordering::SeqCst), 1);
    assert_eq!(receiver_releases.load(Ordering::SeqCst), 1);
    drop(frame);

    assert_eq!(impl_releases.load(Ordering::SeqCst), 1);
    assert_eq!(receiver_releases.load(Ordering::SeqCst), 1);
}
