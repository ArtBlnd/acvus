//! `WordRuntime`'s values and its operations.

use std::any::{TypeId, type_name};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use acvus_extern::Settled;
use acvus_extern::adapter::{Breach, Validate, Validity};
use acvus_extern::callback::{RustFn, RustFns};
use acvus_extern::impls::Family;
use acvus_extern::impls::task::TaskKind;
use acvus_extern::runtime::{Crosses, RefTarget, Runtime};
use acvus_utils::Interner;

use crate::boundary::{self, Kind, WordObject, WordTuple, WordVariant, scalar_kind};

use crate::boundary::Word;

/// The runtime: its interner, and nothing else; every value's storage is its
/// own allocation.
#[derive(Default)]
pub struct WordRuntime {
    interner: Interner,
}

impl Runtime for WordRuntime {
    type Value = Word;
    type Tuple = WordTuple;
    type Object = WordObject;
    type Variant = WordVariant;
    type Shared<'a, T: Crosses + 'a> = &'a T;
    type Exclusive<'a, T: Crosses + 'a> = &'a mut T;
    type SharedSlice<'a, T: Crosses + 'a> = &'a [T];
    type ExclusiveSlice<'a, T: Crosses + 'a> = &'a mut [T];
    type SharedStr<'a> = &'a str;

    fn take<T: Crosses>(&self, v: Word, at: Settled<'_>) -> T {
        boundary::take(v, at)
    }

    fn give<T: Crosses>(&self, value: T, _at: Settled<'_>) -> Word {
        boundary::erase(value)
    }

    fn shared<'a, T: Crosses + 'a>(&'a self, v: &'a Word, at: Settled<'a>) -> &'a T {
        boundary::place(v, at)
    }

    fn exclusive<'a, T: Crosses + 'a>(&'a self, v: &'a mut Word, at: Settled<'a>) -> &'a mut T {
        boundary::place_mut(v, at)
    }

    fn shared_slice<'a, T: Crosses + 'a>(&'a self, v: &'a Word, at: Settled<'a>) -> &'a [T] {
        boundary::run(v, at)
    }

    fn exclusive_slice<'a, T: Crosses + 'a>(&'a self, v: &'a mut Word, at: Settled<'a>) -> &'a mut [T] {
        boundary::run_mut(v, at)
    }

    fn shared_str<'a>(&'a self, v: &'a Word, at: Settled<'a>) -> &'a str {
        boundary::text(v, at)
    }

    fn take_tuple(&self, v: Word, at: Settled<'_>) -> WordTuple {
        boundary::take_tuple(v, at)
    }

    fn take_object(&self, v: Word, at: Settled<'_>) -> WordObject {
        boundary::take_object(v, at)
    }

    fn take_variant(&self, v: Word, at: Settled<'_>) -> WordVariant {
        boundary::take_variant(v, at)
    }

    fn make_tuple(&self, parts: impl ExactSizeIterator<Item = Word>, at: Settled<'_>) -> Word {
        boundary::make_tuple(parts, at)
    }

    fn make_object(&self, fields: impl ExactSizeIterator<Item = Word>, at: Settled<'_>) -> Word {
        boundary::make_object(fields, at)
    }

    fn make_variant(&self, tag: u32, payload: Option<Word>, at: Settled<'_>) -> Word {
        boundary::make_variant(tag, payload, at)
    }

    fn lent_part<'a>(&'a self, v: Word, i: u32, at: Settled<'a>) -> Word {
        boundary::lent_part(v, i, at)
    }

    fn lent_tag(&self, v: Word, at: Settled<'_>) -> u32 {
        boundary::lent_tag(v, at)
    }

    fn reference<'a, P: ?Sized>(&'a self, target: RefTarget<'a, P>, _at: Settled<'a>) -> Word {
        boundary::reference(target)
    }

    fn interner(&self) -> &Interner {
        &self.interner
    }

    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send {
        Deadline(Instant::now() + d)
    }
}

impl RustFns for WordRuntime {
    fn rust_fn<R>(&self, f: RustFn<R>) -> Word
    where
        R: Runtime<Value = Word>,
    {
        boundary::closure(f)
    }
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

/// A breach named by the kind a value has and the type it was read as.
fn read_as<T: ?Sized>(kind: Kind) -> Validity {
    Validity::Breach(Breach { what: format!("a {kind:?} value is read as {}", type_name::<T>()) })
}

/// Whether a value named by a reference to storage outside any slot holds
/// what it is read as cannot be told from the reference.
fn names_a_part(kind: Kind) -> bool {
    matches!(kind, Kind::Part | Kind::WidePart)
}

/// Whether a `Large` value holds a value of the canonical type `canon`.
fn holds<T: ?Sized>(v: Word, canon: TypeId, at: Settled<'_>) -> Validity {
    if boundary::canon_of(v, at) == canon {
        Validity::Valid
    } else {
        read_as::<T>(Kind::Large)
    }
}

impl Validate for WordRuntime {
    fn validate<T: Crosses>(&self, v: &Word, at: Settled<'_>) -> Validity {
        let named = boundary::followed(*v, at);
        match named.kind() {
            kind if names_a_part(kind) => Validity::Unknown,
            Kind::Large => holds::<T>(named, TypeId::of::<T::Canon>(), at),
            kind if kind.is_scalar() && scalar_kind::<T>() == Some(kind) => Validity::Valid,
            kind => read_as::<T>(kind),
        }
    }

    fn validate_run<T: Crosses>(&self, v: &Word, at: Settled<'_>) -> Validity {
        let named = boundary::followed(*v, at);
        match named.kind() {
            kind if names_a_part(kind) => Validity::Unknown,
            Kind::Large => holds::<[T]>(named, TypeId::of::<Vec<T::Canon>>(), at),
            kind => read_as::<[T]>(kind),
        }
    }

    fn validate_text(&self, v: &Word, at: Settled<'_>) -> Validity {
        let named = boundary::followed(*v, at);
        match named.kind() {
            kind if names_a_part(kind) => Validity::Unknown,
            Kind::Large => holds::<str>(named, TypeId::of::<String>(), at),
            kind => read_as::<str>(kind),
        }
    }

    /// No value of this runtime is an impl value until it implements
    /// `Instances`.
    fn validate_instance<F: Family, K: TaskKind>(&self, v: &Word, _at: Settled<'_>) -> Validity {
        read_as::<F>(v.kind())
    }

    fn validate_impl(&self, v: &Word, _at: Settled<'_>) -> Validity {
        Validity::Breach(Breach { what: format!("a {:?} value is read as an impl value", v.kind()) })
    }

    fn validate_closure(&self, v: &Word, _at: Settled<'_>) -> Validity {
        match v.kind() {
            Kind::Closure => Validity::Valid,
            kind => Validity::Breach(Breach { what: format!("a {kind:?} value is called as a closure") }),
        }
    }
}
