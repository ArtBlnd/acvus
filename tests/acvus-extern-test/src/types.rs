//! The extern-defined types of the crossing's tests, each declared by
//! `#[derive(Opaque)]` as an extern crate declares its own.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use std::marker::PhantomData;

use acvus_extern::decl::{EffectVar, IdentityVar, TypeVar};
use acvus_extern::runtime::Runtime;
use acvus_extern::{Opaque, Ref};

/// A value that counts its drops, for the crossing's release tests.
#[derive(Opaque)]
pub struct Tally(pub Arc<AtomicUsize>);

impl Drop for Tally {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// An extern-defined struct, for borrowed results of a field and of an
/// element the struct owns.
#[derive(Opaque)]
pub struct Pair {
    pub name: String,
    pub lead: Number,
    pub items: Vec<Number>,
}

/// An extern-defined number, for a borrowed result that names one.
#[derive(Opaque, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Number(pub i64);

/// An extern-defined type that keeps a loan on a `Pair` an argument lent, as
/// the reference value the argument was: its lifetime is the loan's.
#[derive(Opaque)]
pub struct Cursor<'a, R: Runtime> {
    pub over: Ref<'a, R, Pair>,
    pub at: usize,
}

/// An extern-defined type that carries an effect and an identity, each held as
/// a marker: a session, of one identity, whose work runs at its effect.
#[derive(Opaque)]
pub struct Session<R: Runtime, T: TypeVar<R>, E: EffectVar, I: IdentityVar> {
    pub pending: Vec<T>,
    pub at: PhantomData<fn() -> (R, E, I)>,
}

/// An extern-defined generic type: a stack of a type variable's values, each
/// held as the runtime's uniform value.
#[derive(Opaque)]
pub struct Stack<R: Runtime, T: TypeVar<R>> {
    pub items: Vec<T>,
    pub runtime: PhantomData<fn() -> R>,
}
