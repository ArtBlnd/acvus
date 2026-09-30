//! `WordRuntime`'s values and its operations.


use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use acvus_extern::callback::{CallArgs, CallsBack, LoanForm};
use acvus_extern::{Settled, SiteShape};
use acvus_extern::Primitive;
use acvus_extern::runtime::{Crosses, Mutability, Opaque, RefTarget, Runtime};
use acvus_extern::register::{ExternFn, Refused, Registers};
use acvus_utils::{Interner, QualifiedRef};

use crate::boundary::{self, Held, Word};

/// The runtime: its interner, and nothing else; every value's storage is its
/// own allocation.
#[derive(Default)]
pub struct WordRuntime {
    interner: Interner,
    /// The functions registered at this runtime, by path.
    registered: HashMap<QualifiedRef, Box<dyn ExternFn<WordRuntime>>>,
}

impl Registers for WordRuntime {
    fn register<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self>,
    {
        let path = f.path(&self.interner);
        if self.registered.contains_key(&path) {
            return Err(Refused::PathTaken(path));
        }
        self.registered.insert(path, Box::new(f));
        Ok(())
    }

    fn registered(&self, path: QualifiedRef) -> Option<&dyn ExternFn<Self>> {
        self.registered.get(&path).map(|f| &**f)
    }
}

impl Runtime for WordRuntime {
    type Value = Word;
    type Shared<'a, T: Opaque + 'a> = &'a T;
    type Exclusive<'a, T: Opaque + 'a> = &'a mut T;

    fn give_opaque<T: Opaque>(&self, value: T, _at: Settled<'_>) -> Word {
        boundary::give_opaque(value)
    }

    fn take_opaque<T: Opaque>(&self, v: Word, at: Settled<'_>) -> T {
        boundary::take_opaque(v, at)
    }

    fn give_primitive<P: Primitive>(&self, value: P, _at: Settled<'_>) -> Word {
        boundary::scalar(value)
    }

    fn take_primitive<P: Primitive>(&self, v: Word, at: Settled<'_>) -> P {
        boundary::take_primitive(v, at)
    }

    fn give_string(&self, value: String, _at: Settled<'_>) -> Word {
        boundary::give_string(value)
    }

    fn take_string(&self, v: Word, at: Settled<'_>) -> String {
        boundary::take_string(v, at)
    }

    fn give_option(&self, value: Option<Word>, _at: Settled<'_>) -> Word {
        boundary::give_option(value)
    }

    fn take_option(&self, v: Word, at: Settled<'_>) -> Option<Word> {
        boundary::take_option(v, at)
    }

    fn give_result(&self, value: Result<Word, Word>, _at: Settled<'_>) -> Word {
        boundary::give_result(value)
    }

    fn take_result(&self, v: Word, at: Settled<'_>) -> Result<Word, Word> {
        boundary::take_result(v, at)
    }

    fn give_tuple<const N: usize>(&self, parts: [Word; N], _at: Settled<'_>) -> Word {
        boundary::give_tuple(parts)
    }

    fn take_tuple<const N: usize>(&self, v: Word, at: Settled<'_>) -> [Word; N] {
        boundary::take_tuple(v, at)
    }

    fn shared<'a, T: Opaque + 'a>(&'a self, v: &'a Word, at: Settled<'a>) -> &'a T {
        boundary::place(v, at)
    }

    fn exclusive<'a, T: Opaque + 'a>(&'a self, v: &'a mut Word, at: Settled<'a>) -> &'a mut T {
        boundary::place_mut(v, at)
    }

    fn reference<'a, P: ?Sized>(&'a self, target: RefTarget<'a, P>, _at: Settled<'a>) -> Word {
        boundary::reference(target)
    }

    fn target<'l, P: ?Sized>(&'l self, v: Word, at: Settled<'l>) -> &'l P {
        boundary::referent(v, at)
    }

    fn target_mut<'l, P: ?Sized>(&'l self, v: Word, at: Settled<'l>) -> &'l mut P {
        boundary::referent_mut(v, at)
    }

    fn interner(&self) -> &Interner {
        &self.interner
    }

    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send {
        Deadline(Instant::now() + d)
    }
}

impl CallsBack for WordRuntime {
    fn call_now<'l, R, A, U>(&self, rt: &'l R, closure: Word, args: A::Lent<'l>, at: Settled<'l>) -> U::At<'l>
    where
        R: Runtime<Value = Word>,
        A: CallArgs<R>,
        U: LoanForm<R>,
    {
        let run = A::lend_run(args, rt, at).map(Held::kept).collect();
        let result = call_closure(rt, closure, run, A::lent().collect(), at);
        U::read(result, rt, at)
    }

    fn call_later<'a, R, A, U>(
        &self,
        rt: &'a R,
        closure: Word,
        args: A::Moved,
        at: Settled<'a>,
    ) -> impl Future<Output = U> + Send
    where
        R: Runtime<Value = Word>,
        A: CallArgs<R>,
        U: Crosses<R>,
    {
        let run: Vec<Held> = A::move_run(args, rt, at).map(Held::kept).collect();
        let lent: Box<[Option<Mutability>]> = A::lent().collect();
        async move {
            let result = call_closure(rt, closure, run, lent, at);
            <U as LoanForm<R>>::read(result, rt, at)
        }
    }
}

/// Calls the Rust function `closure` is with the argument run `run`, each
/// passed as `lent` says, and gives its result. A value given is the callee's
/// from here on; a reference value lent is the caller's, released after the
/// call.
fn call_closure<'a, R>(
    rt: &'a R,
    closure: Word,
    run: Vec<Held>,
    lent: Box<[Option<Mutability>]>,
    at: Settled<'a>,
) -> Word
where
    R: Runtime<Value = Word>,
{
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

    let done = boundary::callee::<R>(closure, at).call(rt, &values, &site);

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

