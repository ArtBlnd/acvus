// expect: lifetime may not live long enough
use lq::*;
pub type Flip<A, B> = (B, A);
pub fn check<'x, 'y>() { same::<Cons<Lt<'x>, Cons<Lt<'y>, Nil>>, <Flip<Ref<'x, Var>, Ref<'y, Var>> as Lifetimed>::Seq>(); }
