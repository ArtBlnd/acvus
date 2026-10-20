// expect: pass
// A kept function value crosses by the impl over every variable, since it is
// a `Uniform`, so Rust admits giving one back. That an extern gives acvus no
// function value (functions) is kept at registration, where a declaration
// whose result holds a function type is refused, not by the types here.
use std::marker::PhantomData;

use model::{Cross, Runtime, Uniform};

pub struct Pred<'f, T>(PhantomData<(fn(T) -> T, fn() -> &'f ())>);

pub fn give_back<R, T>(kept: Uniform<R, Pred<'static, T>>) -> R::Value
where
    R: Runtime,
{
    <Uniform<R, Pred<'static, T>> as Cross<R>>::give(kept)
}
