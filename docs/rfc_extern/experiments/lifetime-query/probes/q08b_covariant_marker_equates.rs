// expect: lifetime may not live long enough
// The other direction of q08a.
use lq::*;
pub fn check<'x, 'y>(_: &'y &'x ()) { same::<Cons<CovLt<'y>, Nil>, Cons<CovLt<'x>, Nil>>(); }
