// expect: E0509
// The cost of the derive's `Drop`: an extern body reads a declared
// structure's fields in place, and cannot move them out of it.
use model::{Runtime, Uniform};

pub struct Point<R>
where
    R: Runtime,
{
    pub x: Uniform<R, i64>,
    pub y: Uniform<R, String>,
}

impl<R> Drop for Point<R>
where
    R: Runtime,
{
    fn drop(&mut self) {}
}

pub fn split<R>(point: Point<R>) -> (Uniform<R, i64>, Uniform<R, String>)
where
    R: Runtime,
{
    let Point { x, y } = point;
    (x, y)
}
