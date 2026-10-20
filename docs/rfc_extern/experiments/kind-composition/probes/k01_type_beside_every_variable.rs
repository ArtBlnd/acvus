// expect: pass
// An author's type crosses by an impl of its own, beside the impl over every
// variable, and nests inside an option and a tuple: coherence admits it, since
// no type but a `Uniform` is a `TypeVar` (variables §3).
use model::{Cross, Runtime};

pub struct Point {
    pub x: i64,
    pub y: i64,
}

// SAFETY: a point crosses as the tuple of its fields.
unsafe impl<R> Cross<R> for Point
where
    R: Runtime,
{
    fn give(self) -> R::Value {
        <(i64, i64) as Cross<R>>::give((self.x, self.y))
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        let (x, y) = unsafe { <(i64, i64) as Cross<R>>::take(value) };
        Point { x, y }
    }
}

pub fn nested<R>(point: Point) -> R::Value
where
    R: Runtime,
{
    <Option<(Point, Option<i64>)> as Cross<R>>::give(Some((point, None)))
}
