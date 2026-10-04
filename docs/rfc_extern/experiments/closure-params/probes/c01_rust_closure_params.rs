// expect: pass
// Author code with Rust's own closure types, generic over type variables.
use model::{Never, Ref, TypeVar, Uniform};

pub fn find<'x, T: TypeVar>(xs: &'x [T], pred: impl Fn(&T) -> bool) -> Option<Ref<'x, T>> {
    xs.iter().find(|x| pred(x)).map(|x| x.as_ref_var())
}

// The result borrows the call's argument, by elision.
pub fn map_first<'x, T: TypeVar, U: TypeVar>(xs: &'x [T], f: impl Fn(&T) -> Ref<'_, U>) -> Option<Ref<'x, U>> {
    xs.first().map(|x| f(x))
}

// Two arguments, the result borrowing the first alone: exact per argument.
pub fn pick<'x, T: TypeVar>(a: &'x T, b: &T, f: impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, T>) -> Ref<'x, T> {
    f(a, b)
}

pub fn each_mut<T: TypeVar>(xs: &mut [T], mut f: impl FnMut(&mut T)) {
    xs.iter_mut().for_each(|x| f(x));
}

// An entry, monomorphic at `Uniform<Never>`, handing each body a closure it
// writes itself; here the closures stand for calls into an acvus function value.
pub fn entry(xs: &mut [Uniform<Never>], v: &Uniform<Never>) {
    let _ = find(xs, |x| { let _ = x.as_ref_var(); true });
    let _ = map_first::<Uniform<Never>, Uniform<Never>>(xs, |x| x.as_ref_var());
    let _ = pick(v, v, |p, _| p.as_ref_var());
    each_mut(xs, |_x| {});
}
