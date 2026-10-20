// expect: pass
// A generic and a concrete instantiation of one type differ only in a
// zero-sized marker (variables §3, §4): their sizes and alignments agree.
use model::*;

pub struct Filter<'f, R, It, T, E>
where
    R: Runtime,
    It: TypeVar<R>,
    T: TypeVar<R>,
    E: EffectVar,
{
    pub source: Instance<R, Next, It, (It, T, E)>,
    pub pred: Uniform<R, Pred<'f, T, E>>,
}

type GenericView = Filter<'static, Probe, Uniform<Probe, Generic>, Uniform<Probe, Generic>, Effect0>;
type ConcreteView = Filter<'static, Probe, Uniform<Probe, Vec<i64>>, Uniform<Probe, String>, Effect0>;

const _: () = assert!(core::mem::size_of::<GenericView>() == core::mem::size_of::<ConcreteView>());
const _: () = assert!(core::mem::align_of::<GenericView>() == core::mem::align_of::<ConcreteView>());
