// expect: pass
// What `#[fn_type(ref)] type Pred<'f, T, E: EffectVar> = fn(&T) -> bool;` would
// expand to, in the author's crate: a marker carrying the function's effect
// variable, and a local trait that projects `Uniform` of it to Rust's own
// closure, its signature written as the alias wrote it.
use std::marker::PhantomData;
use model::{Effect0, EffectVar, Never, Ref, TypeVar, Uniform};

pub struct Pred<'f, T, E>(PhantomData<(fn(T) -> T, fn() -> &'f (), fn(E) -> E)>);
pub trait PredFn<T> { fn as_fn<'s>(&'s self) -> impl Fn(&T) -> bool + 's; }
impl<'f, T: TypeVar, E: EffectVar> PredFn<T> for Uniform<Pred<'f, T, E>> {
    fn as_fn<'s>(&'s self) -> impl Fn(&T) -> bool + 's {
        move |x: &T| { let _ = (self.bits(), x.as_ref_var()); true } // stands for a call into the function value
    }
}

pub struct Pick<'f, T, E>(PhantomData<(fn(T) -> T, fn() -> &'f (), fn(E) -> E)>);
pub trait PickFn<T> { fn as_fn<'s>(&'s self) -> impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, T> + 's; }
impl<'f, T: TypeVar, E: EffectVar> PickFn<T> for Uniform<Pick<'f, T, E>> {
    fn as_fn<'s>(&'s self) -> impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, T> + 's {
        move |p: &T, _q: &T| { let _ = self.bits(); p.as_ref_var() }
    }
}

// An author's type keeping a function past the call, and its use later.
pub struct Filter<'f, T, E> { pub pred: Uniform<Pred<'f, T, E>>, pub seen: u64 }
pub fn keep<'f, T: TypeVar, E: EffectVar>(pred: Uniform<Pred<'f, T, E>>) -> Filter<'f, T, E> { Filter { pred, seen: 0 } }
pub fn count<T: TypeVar, E: EffectVar>(filter: &mut Filter<'_, T, E>, xs: &[T]) -> u64 {
    let pred = filter.pred.as_fn();
    let n = xs.iter().filter(|x| pred(x)).count() as u64;
    filter.seen += n;
    n
}
pub fn pick_with<'x, T: TypeVar, E: EffectVar>(f: &Uniform<Pick<'_, T, E>>, a: &'x T, b: &T) -> Ref<'x, T> { (f.as_fn())(a, b) }

// The same stored function seen concretely, and at the generic instantiation.
pub fn concrete(filter: &mut Filter<'_, Uniform<String>, Effect0>, xs: &[Uniform<String>]) -> u64 { count(filter, xs) }
pub fn generic(filter: &mut Filter<'_, Uniform<Never>, Effect0>, xs: &[Uniform<Never>]) -> u64 { count(filter, xs) }
