// expect: E0277
// The same fields with no bound on the parameters: the fields are not
// well formed, so the bounds must be written for the author.
use model::*;

pub struct Filter<'f, R, It, T, E> {
    pub source: Instance<R, Next, It, (It, T, E)>,
    pub pred: Uniform<R, Pred<'f, T, E>>,
}
