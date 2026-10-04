// expect: lifetime may not live long enough
// The control: covariant markers at one position, with 'x: 'y implied. A
// subtyping check would admit this direction or the other (q08b); the query,
// put through trait matching, equates its sides and refuses both.
use lq::*;
pub fn check<'x, 'y>(_: &'y &'x ()) { same::<Cons<CovLt<'x>, Nil>, Cons<CovLt<'y>, Nil>>(); }
