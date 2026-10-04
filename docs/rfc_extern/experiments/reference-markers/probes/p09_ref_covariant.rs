// expect: pass
use model::Ref;
pub fn shorten<'x, 'a: 'x>(v: &'x Vec<Ref<'a, i64>>) -> &'x Vec<Ref<'x, i64>> { v }
