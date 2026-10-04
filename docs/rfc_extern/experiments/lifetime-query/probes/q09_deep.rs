// expect: pass
// Sixteen lifetimes, four levels of Zip with Ref chains inside.
use lq::*;
pub fn check<'l0,'l1,'l2,'l3,'l4,'l5,'l6,'l7,'l8,'l9,'l10,'l11,'l12,'l13,'l14,'l15>() {
    same::<Cons<Lt<'l0>, Cons<Lt<'l1>, Cons<Lt<'l2>, Cons<Lt<'l3>, Cons<Lt<'l4>, Cons<Lt<'l5>, Cons<Lt<'l6>, Cons<Lt<'l7>, Cons<Lt<'l7>, Cons<Lt<'l8>, Cons<Lt<'l9>, Cons<Lt<'l10>, Cons<Lt<'l11>, Cons<Lt<'l12>, Cons<Lt<'l13>, Cons<Lt<'l14>, Cons<Lt<'l15>, Nil>>>>>>>>>>>>>>>>>, <Zip<'l0, 'l1, Zip<'l2, 'l3, Ref<'l4, Ref<'l5, Var>>, Ref<'l6, Var>>, Zip<'l7, 'l7, Zip<'l8, 'l9, Zip<'l10, 'l11, Ref<'l12, Ref<'l13, Var>>, Ref<'l14, Var>>, Ref<'l15, Var>>, Var>> as Lifetimed>::Seq>();
}
