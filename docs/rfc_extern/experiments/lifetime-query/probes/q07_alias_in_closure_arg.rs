// expect: lifetime may not live long enough
use lq::*;
pub type Swap<'a, 'b> = Pair<'b, 'a>;
// The syntax reads `&'p Swap<'p, 'q>` as 'p, 'p, 'q; the resolved type is `&'p Pair<'q, 'p>`.
pub fn check<'p, 'q>() { same::<Cons<Lt<'p>, Cons<Lt<'p>, Cons<Lt<'q>, Nil>>>, <&'p Swap<'p, 'q> as Lifetimed>::Seq>(); }
