// expect: pass
// What `#[fn_type] type Pred<'f, T> = dyn Fn(&T) -> bool + 'f;` would expand to,
// in the author's crate: a marker, and a local trait that views `Uniform` of
// it as Rust's own closure, its signature written as the alias wrote it.
use std::marker::PhantomData;
use model::{Never, Ref, TypeVar, Uniform};

pub struct Pred<'f, T>(PhantomData<(fn(T) -> T, fn() -> &'f ())>);
pub trait PredFn<T> { fn as_fn<'s>(&'s self) -> impl Fn(&T) -> bool + 's; }
impl<'f, T: TypeVar> PredFn<T> for Uniform<Pred<'f, T>> {
    fn as_fn<'s>(&'s self) -> impl Fn(&T) -> bool + 's {
        move |x: &T| { let _ = (self.bits(), x.as_ref_var()); true } // stands for a call into the function value
    }
}

pub struct Pick<'f, T>(PhantomData<(fn(T) -> T, fn() -> &'f ())>);
pub trait PickFn<T> { fn as_fn<'s>(&'s self) -> impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, T> + 's; }
impl<'f, T: TypeVar> PickFn<T> for Uniform<Pick<'f, T>> {
    fn as_fn<'s>(&'s self) -> impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, T> + 's {
        move |p: &T, _q: &T| { let _ = self.bits(); p.as_ref_var() }
    }
}

// An author's type keeping a function past the call, and its use later.
pub struct Filter<'f, T> { pub pred: Uniform<Pred<'f, T>>, pub seen: u64 }
pub fn keep<'f, T: TypeVar>(pred: Uniform<Pred<'f, T>>) -> Filter<'f, T> { Filter { pred, seen: 0 } }
pub fn count<T: TypeVar>(filter: &mut Filter<'_, T>, xs: &[T]) -> u64 {
    let pred = filter.pred.as_fn();
    let n = xs.iter().filter(|x| pred(x)).count() as u64;
    filter.seen += n;
    n
}
pub fn pick_with<'x, T: TypeVar>(f: &Uniform<Pick<'_, T>>, a: &'x T, b: &T) -> Ref<'x, T> { (f.as_fn())(a, b) }

// The same stored function seen concretely, and at the generic view.
pub fn concrete(filter: &mut Filter<'_, Uniform<String>>, xs: &[Uniform<String>]) -> u64 { count(filter, xs) }
pub fn generic(filter: &mut Filter<'_, Uniform<Never>>, xs: &[Uniform<Never>]) -> u64 { count(filter, xs) }
