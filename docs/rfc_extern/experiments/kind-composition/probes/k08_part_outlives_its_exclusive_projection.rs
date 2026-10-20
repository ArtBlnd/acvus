// expect: E0521
// An exclusive reference to a part, reborrowed, does not leave the projection of
// the tuple it was opened in, so two exclusive references to one part never
// meet.
use model::{Mut, Runtime, Uniform};

pub fn escape<'a, R>(pair: &'a mut Uniform<R, (i64, i64)>, out: &mut Option<Mut<'a, R, i64>>)
where
    R: Runtime,
{
    pair.project_mut(|(first, _)| {
        *out = Some(first.as_mut());
    });
}
