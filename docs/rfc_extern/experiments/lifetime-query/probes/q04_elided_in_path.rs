// expect: E0277
// A lifetime elided in a path, which the syntax does not show.
use lq::*;
pub fn check() { same::<Nil, <Cursor<Var> as Lifetimed>::Seq>(); }
