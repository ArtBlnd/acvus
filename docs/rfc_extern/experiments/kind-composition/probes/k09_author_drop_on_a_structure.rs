// expect: E0119
// The derive writes an empty `Drop` for a declared structure, so an author's
// own is refused: a declared structure's release is its fields' alone.
use model::{Runtime, Uniform};

pub struct Point<R>
where
    R: Runtime,
{
    pub x: Uniform<R, i64>,
}

// What the derive writes.
impl<R> Drop for Point<R>
where
    R: Runtime,
{
    fn drop(&mut self) {}
}

// What an author would write.
impl<R> Drop for Point<R>
where
    R: Runtime,
{
    fn drop(&mut self) {}
}
