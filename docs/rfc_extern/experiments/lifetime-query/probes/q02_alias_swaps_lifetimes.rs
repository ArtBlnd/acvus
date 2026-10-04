// expect: lifetime may not live long enough
use lq::*;
pub type Swap<'a, 'b> = Pair<'b, 'a>;
pub fn check<'x, 'y>() { same::<Cons<Lt<'x>, Cons<Lt<'y>, Nil>>, <Swap<'x, 'y> as Lifetimed>::Seq>(); }
