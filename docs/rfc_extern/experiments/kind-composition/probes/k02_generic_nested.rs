// expect: pass
// Generic code gives and takes back nestings of its variables beside concrete
// parts, each layer by its own kind.
use model::{Cross, Runtime, TypeVar};

pub fn give<R, T, U>(value: Result<Option<T>, (U, Option<i64>)>) -> R::Value
where
    R: Runtime,
    T: TypeVar<R>,
    U: TypeVar<R>,
{
    <Result<Option<T>, (U, Option<i64>)> as Cross<R>>::give(value)
}

pub fn take<R, T, U>(value: R::Value) -> Option<Option<(U, Result<T, bool>)>>
where
    R: Runtime,
    T: TypeVar<R>,
    U: TypeVar<R>,
{
    // SAFETY: the caller gives a value of this type it owns.
    unsafe { <Option<Option<(U, Result<T, bool>)>> as Cross<R>>::take(value) }
}
