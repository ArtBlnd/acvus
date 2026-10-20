// expect: pass
// The queries a macro poses for
//   fn f<'x, 'y, 'f, T, E>(xs: &'x [T], c: Cursor<'y, Uniform<Shared<'x, T>>>, p: Uniform<Pred<'f, T, E>>,
//                          z: Zip<'x, 'y, Ref<'x, T>, Mut<'y, T>>, s: S<'x, 'y>) -> Option<Ref<'x, T>>
// with its type variable put as `Var`, and its effect variable left as
// itself, since it carries no lifetime.
use lq::*;
pub fn check<'x, 'y, 'f, E>() {
    same::<Cons<Lt<'x>, Nil>, <&'x [Var] as Lifetimed>::Seq>();
    same::<Cons<Lt<'y>, Cons<Lt<'x>, Nil>>, <Cursor<'y, Uniform<Shared<'x, Var>>> as Lifetimed>::Seq>();
    same::<Cons<Lt<'f>, Nil>, <Uniform<Pred<'f, Var, E>> as Lifetimed>::Seq>();
    same::<Cons<Lt<'x>, Cons<Lt<'y>, Cons<Lt<'x>, Cons<Lt<'y>, Nil>>>>, <Zip<'x, 'y, Ref<'x, Var>, Mut<'y, Var>> as Lifetimed>::Seq>();
    same::<Cons<Lt<'x>, Cons<Lt<'y>, Nil>>, <S<'x, 'y> as Lifetimed>::Seq>();
    same::<Cons<Lt<'x>, Nil>, <Option<Ref<'x, Var>> as Lifetimed>::Seq>();
}
