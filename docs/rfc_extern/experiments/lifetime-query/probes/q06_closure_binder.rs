// expect: pass
// The queries for a closure parameter's own signature,
//   impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, T>,
// its binder's lifetimes standing as the check's own.
use lq::*;
pub fn check<'p, 'q>() {
    same::<Cons<Lt<'p>, Nil>, <&'p Var as Lifetimed>::Seq>();
    same::<Cons<Lt<'q>, Nil>, <&'q Var as Lifetimed>::Seq>();
    same::<Cons<Lt<'p>, Nil>, <Ref<'p, Var> as Lifetimed>::Seq>();
}
