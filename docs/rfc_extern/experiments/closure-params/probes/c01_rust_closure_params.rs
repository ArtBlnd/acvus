// expect: pass
// Author code with Rust's own closure types, generic over type variables, each
// closure's effect the declaration's effect variable `E`.
use model::{Effect0, EffectVar, Never, Ref, TypeVar, Uniform};

pub fn find<'x, T: TypeVar, E: EffectVar>(xs: &'x [T], pred: impl Fn(&T) -> bool) -> Option<Ref<'x, T>> {
    xs.iter().find(|x| pred(x)).map(|x| x.as_ref_var())
}

// The result borrows the call's argument, by elision.
pub fn map_first<'x, T: TypeVar, U: TypeVar, E: EffectVar>(xs: &'x [T], f: impl Fn(&T) -> Ref<'_, U>) -> Option<Ref<'x, U>> {
    xs.first().map(|x| f(x))
}

// Two arguments, the result borrowing the first alone: exact per argument.
pub fn pick<'x, T: TypeVar, E: EffectVar>(a: &'x T, b: &T, f: impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, T>) -> Ref<'x, T> {
    f(a, b)
}

pub fn each_mut<T: TypeVar, E: EffectVar>(xs: &mut [T], mut f: impl FnMut(&mut T)) {
    xs.iter_mut().for_each(|x| f(x));
}

// An entry, monomorphic at `Uniform<Never>` and `Effect0`, handing each body a
// closure it writes itself; here the closures stand for calls into an acvus
// function value.
pub fn entry(xs: &mut [Uniform<Never>], v: &Uniform<Never>) {
    let _ = find::<Uniform<Never>, Effect0>(xs, |x| { let _ = x.as_ref_var(); true });
    let _ = map_first::<Uniform<Never>, Uniform<Never>, Effect0>(xs, |x| x.as_ref_var());
    let _ = pick::<Uniform<Never>, Effect0>(v, v, |p, _| p.as_ref_var());
    each_mut::<Uniform<Never>, Effect0>(xs, |_x| {});
}
