//! `WordRuntime`'s values and its operations.

use std::collections::HashSet;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use acvus_extern::callback::{CallArgs, CallsBack, LoanForm, MovedArgs};
use acvus_extern::{Settled, SiteShape};
use acvus_extern::runtime::{Crosses, Mutability, Runtime};
use acvus_extern::register::{AwaitedFn, Declared, ExternFn, Refused, Registers};
use acvus_utils::{Interner, QualifiedRef};

use crate::boundary::{self, Held, Kinds, Word};

/// The runtime: its interner and the paths registered at it; every value's
/// storage is its own allocation. It runs no compiled program, so it keeps no
/// registered function: the tests call a function through its dispatch.
#[derive(Default)]
pub struct WordRuntime {
    interner: Interner,
    registered: HashSet<QualifiedRef>,
}

impl WordRuntime {
    /// Registers `path`, refused where a function is registered there.
    fn enter(&mut self, path: QualifiedRef) -> Result<(), Refused> {
        if !self.registered.insert(path) {
            return Err(Refused::PathTaken(path));
        }
        Ok(())
    }
}

impl Registers for WordRuntime {
    fn register<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self> + Declared<Self>,
    {
        self.enter(Declared::path(&f, &self.interner))
    }

    fn register_awaited<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: AwaitedFn<Self> + Declared<Self>,
    {
        self.enter(Declared::path(&f, &self.interner))
    }
}

impl Runtime for WordRuntime {
    type Value = Word;

    type Opaque = Kinds;
    type Primitive = Kinds;
    type String = Kinds;
    type Option = Kinds;
    type Result = Kinds;
    type Tuple = Kinds;
    type Ref = Kinds;

    fn interner(&self) -> &Interner {
        &self.interner
    }

    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send {
        Deadline(Instant::now() + d)
    }
}

impl CallsBack for WordRuntime {
    fn call_now<'l, A, U>(&'l self, closure: Word, args: A::Lent<'l>, at: Settled<'l>) -> U::At<'l>
    where
        A: CallArgs<Self> + 'l,
        U: LoanForm<Self> + 'l,
    {
        let run = A::lend_run(args, self, at).map(Held::kept).collect();
        let result = call_closure(self, closure, run, A::lent().collect(), at);
        U::read(result, self, at)
    }

    fn call_later<'a, A, U>(
        &'a self,
        closure: Word,
        args: A::Moved,
        at: Settled<'a>,
    ) -> impl Future<Output = U> + Send
    where
        A: MovedArgs<Self>,
        U: Crosses<Self>,
    {
        let run: Vec<Held> = A::move_run(args, self, at).map(Held::kept).collect();
        let lent: Box<[Option<Mutability>]> = A::lent().collect();
        async move {
            let result = call_closure(self, closure, run, lent, at);
            U::take(self, result, at)
        }
    }
}

/// Calls the Rust function `closure` is with the argument run `run`, each
/// passed as `lent` says, and gives its result. A value given is the callee's
/// from here on; a reference value lent is the caller's, released after the
/// call.
fn call_closure<'a>(
    rt: &'a WordRuntime,
    closure: Word,
    run: Vec<Held>,
    lent: Box<[Option<Mutability>]>,
    at: Settled<'a>,
) -> Word {
    let mut values = Vec::with_capacity(run.len());
    let mut references = Vec::new();
    for (held, passed) in run.into_iter().zip(&lent) {
        match passed {
            Some(_) => {
                values.push(held.same());
                references.push(held);
            }
            None => values.push(held.hand_over()),
        }
    }
    let site = SiteShape::new(lent, Box::new([]));

    let done = boundary::callee(closure, at).call(rt, &values, &site);

    drop(references);
    done.into_value()
}


/// A wait until an instant. This runtime owns no timer and no executor, so
/// the future asks to be polled again until the instant has passed; the
/// crossing's tests wait for short times only, and a timer would bring a
/// thread or a dependency this runtime otherwise does without.
struct Deadline(Instant);

impl Future for Deadline {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.0 {
            return Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

