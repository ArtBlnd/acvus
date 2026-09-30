//! An impl value reaches only the receiver it was chosen for (G6, K14).

use std::marker::PhantomData;

use crate::impls::receiver::{Reads, Steps};
use crate::context::{Awaited, Returning};
use crate::impls::{Family, Instances};

/// A requirement of a generic function that takes its receiver by value or
/// `&mut`: the receiver and the impl chosen for it at its site, owned as one.
/// Only the entry makes one, and no API separates the two.
pub struct Instance<R: Instances, F: Family<Receiver = Steps>, I> {
    receiver: I,
    entry: R::Value,
    _family: PhantomData<fn() -> F>,
}

/// A requirement of a generic function that takes its receiver by `&`: the
/// impl chosen at the types `Ts`, stating the law `L`.
pub struct InstanceOf<R: Instances, F: Family<Receiver = Reads>, Ts, L = ()> {
    entry: R::Value,
    _at: PhantomData<fn() -> (F, Ts, L)>,
}

impl<R: Instances, F: Family<Receiver = Steps>, I> Instance<R, F, I> {
    pub fn call<'r>(&'r mut self, cx: &Returning<'_, R>, rest: F::Rest<'r, R>) -> F::Output<'r, R> {
        todo!("read the entry at the returning task, call its glue on this receiver")
    }

    pub fn call_await<'r>(
        &'r mut self,
        cx: &Awaited<'_, R>,
        rest: F::RestMoved<R>,
    ) -> impl Future<Output = F::Output<'r, R>> {
        async { todo!("read the entry's task, call its glue on this receiver") }
    }

    /// The receiver back, the impl dropped with the pairing.
    pub fn into_receiver(self) -> I {
        self.receiver
    }
}

impl<R: Instances, F: Family<Receiver = Reads>, Ts, L> InstanceOf<R, F, Ts, L> {
    pub fn call<'r>(
        &self,
        cx: &Returning<'_, R>,
        receiver: &'r F::ReceiverAt<Ts>,
        rest: F::Rest<'r, R>,
    ) -> F::Output<'r, R> {
        todo!("read the entry at the returning task, call its glue")
    }

    pub fn call_await<'r>(
        &self,
        cx: &Awaited<'_, R>,
        receiver: F::ReceiverAt<Ts>,
        rest: F::RestMoved<R>,
    ) -> impl Future<Output = F::Output<'r, R>> {
        async { todo!("read the entry's task, call its glue") }
    }
}
