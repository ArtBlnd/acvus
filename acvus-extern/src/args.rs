//! A call's arguments at the declaration's own type variables, sealed
//! (RFC-0097 rule 1).
//!
//! A handler that names a position only as a type variable cannot read it as
//! a Rust type: the glue fills the variable with `Owned`, which no body
//! opens (RFC-0023 rule 10). `Args` holds such positions with the type the
//! checker settled for each at the call site, and lends one to a closure
//! whose parameter type is checked against that settled type first, as
//! `lend` checks a host's closure. It hands out no word, no `Ty` and no
//! value.
//!
//! What the closure borrows does not outlive it,
//!
//! ```compile_fail
//! # use acvus_extern::{Args, Owned, Runtime};
//! fn kept<'c, R: Runtime>(args: &'c Args<'_, (Owned<R>,), R>) -> &'c String {
//!     let mut kept = None;
//!     args.with(0, |s: &String| kept = Some(s));
//!     kept.unwrap()
//! }
//! ```
//!
//! and a write through `with_mut` needs the view exclusively:
//!
//! ```compile_fail
//! # use acvus_extern::{Args, Owned, Runtime};
//! fn shared<R: Runtime>(args: &Args<'_, (Owned<R>,), R>) {
//!     args.with_mut(0, |n: &mut i64| *n += 1);
//! }
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_mir::ty::{Mutability, Ty, TypeArg};
use acvus_utils::Interner;

use crate::crossing::{Crossing, Holding};
use crate::handler::{Arg, ArgRun, CallSite, Takes};
use crate::lend::{Borrows, Lendable, lend, lend_through};
use crate::loan::{Mut, Shared};
use crate::obj::{Form, FormKind, SurvivesSuspension};
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::space::SpaceError;

mod sealed {
    pub trait Sealed {}
}

/// The positions an `Args` covers, as the tuple of the declaration's type
/// variables the handler writes: each member is one acvus parameter taken
/// by value, one word of the call's argument run.
pub trait Members: sealed::Sealed {
    const LEN: usize;

    type Words<Rt>: AsRef<[Owned<Rt>]> + AsMut<[Owned<Rt>]> + Send + Sync
    where
        Rt: Runtime;
    type Onto<Run>: ArgRun
    where
        Run: ArgRun;

    /// # Safety
    /// `run` is `LEN` words the call moved to its callee, and no other
    /// holder owns any of them.
    unsafe fn moved_in<Rt>(holding: Holding<'_, Rt>, run: &[Rt::Value]) -> Self::Words<Rt>
    where
        Rt: Runtime;
}

macro_rules! onto {
    ($run:ty) => { $run };
    ($run:ty, $member:ident $(, $rest:ident)*) => {
        onto!(<$run as ArgRun>::WithOne $(, $rest)*)
    };
}

macro_rules! members {
    ($len:literal: $($member:ident),+) => {
        impl<$($member,)+> sealed::Sealed for ($($member,)+) {}

        impl<$($member,)+> Members for ($($member,)+) {
            const LEN: usize = $len;

            type Words<Rt>
                = [Owned<Rt>; $len]
            where
                Rt: Runtime;
            type Onto<Run>
                = onto!(Run $(, $member)+)
            where
                Run: ArgRun;

            unsafe fn moved_in<Rt>(holding: Holding<'_, Rt>, run: &[Rt::Value]) -> [Owned<Rt>; $len]
            where
                Rt: Runtime,
            {
                // SAFETY: the caller's contract: each word was moved to the
                // callee and no other holder owns it.
                std::array::from_fn(|at| unsafe { Owned::from_value(holding, run[at]) })
            }
        }
    };
}

members!(1: A0);
members!(2: A0, A1);
members!(3: A0, A1, A2);
members!(4: A0, A1, A2, A3);
members!(5: A0, A1, A2, A3, A4);
members!(6: A0, A1, A2, A3, A4, A5);
members!(7: A0, A1, A2, A3, A4, A5, A6);
members!(8: A0, A1, A2, A3, A4, A5, A6, A7);

pub const MOST_MEMBERS: usize = <(u8, u8, u8, u8, u8, u8, u8, u8) as Members>::LEN;

/// The run an `Args` takes: one word per member.
pub struct Positions<P>(PhantomData<fn() -> P>);

impl<P> Form for Positions<P>
where
    P: Members,
{
    const WIDTH: usize = P::LEN;
    const KIND: FormKind = FormKind::Components;

    type Onto<Run>
        = P::Onto<Run>
    where
        Run: ArgRun;
}

/// Each member is a word the call moved to its callee, and the view owns
/// it, so the run survives the caller suspending as a by-value word does.
impl<P> SurvivesSuspension for Positions<P> where P: Members {}

/// The marker `#[extern_fn]` writes for an `Args` parameter.
pub struct ByArgs<P>(PhantomData<fn() -> P>);

/// What an `Args` reads at its call site: the settled type of each member
/// and the interner they were settled under, owned by the site table.
#[derive(Clone)]
pub struct ArgsSite {
    interner: Interner,
    tys: Arc<[Ty]>,
}

impl ArgsSite {
    pub(crate) fn new(interner: Interner, tys: Arc<[Ty]>) -> Self {
        ArgsSite { interner, tys }
    }

    pub(crate) fn len(&self) -> usize {
        self.tys.len()
    }
}

impl<P, Rt> Arg<Rt> for ByArgs<P>
where
    P: Members + 'static,
    Rt: Runtime,
{
    type Site = ArgsSite;
    type Form = Positions<P>;

    const ARGUMENTS: usize = P::LEN;
    const LENDS_A_WORD: bool = false;

    fn site(site: &CallSite<'_, Rt>, at: usize) -> ArgsSite {
        let members = &site.args[at..at + P::LEN];
        ArgsSite {
            interner: members[0].interner.clone(),
            tys: members.iter().map(|member| member.ty.clone()).collect(),
        }
    }

    #[inline(always)]
    unsafe fn loan_ended(_: &Rt, _: &[Rt::Value], _: &ArgsSite) {}
}

/// The canonical bytes `Args::encode` lays the arguments out as (RFC-0033).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded(Box<[u8]>);

impl Encoded {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn into_bytes(self) -> Box<[u8]> {
        self.0
    }
}

/// The call's arguments at the positions `P` names, with the type the
/// checker settled for each at this call site.
pub struct Args<'call, P, Rt>
where
    P: Members,
    Rt: Runtime,
{
    rt: &'call Rt,
    site: &'call ArgsSite,
    words: P::Words<Rt>,
}

impl<'call, P, Rt> Args<'call, P, Rt>
where
    P: Members,
    Rt: Runtime,
{
    /// # Safety
    /// `run` is `P::LEN` words the call moved to its callee, each crossed at
    /// the type `site` holds at its position, and no other holder owns them.
    pub(crate) unsafe fn moved_in(rt: Crossing<'call, Rt>, run: &[Rt::Value], site: &'call ArgsSite) -> Self {
        debug_assert_eq!(run.len(), P::LEN, "an Args run is one word per member");
        Args {
            rt: rt.rt(),
            site,
            // SAFETY: the caller's contract.
            words: unsafe { P::moved_in(rt.holding(), run) },
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        P::LEN
    }

    /// `f` lent argument `i`, or its target where the argument is a
    /// reference; `None` where `i` is past the end or `f`'s parameter type
    /// is not the one the checker settled there.
    pub fn with<Q, O, F>(&self, i: usize, f: F) -> Option<O>
    where
        F: Borrows<Rt, Q, O>,
        F::Marker: Lendable<Rt, Loan = Shared>,
    {
        let word: &Rt::Value = self.words.as_ref().get(i)?;
        let interner = &self.site.interner;
        match &self.site.tys[i] {
            Ty::Ref(_, target) => {
                let TypeArg::Uniform(target) = &**target else {
                    return None;
                };
                // SAFETY: the word was crossed at `&target`, so it is a
                // reference to a live storage of `target` the caller keeps for
                // the call, and a shared parameter writes nothing.
                unsafe { lend_through(self.rt, interner, word, target, f) }.ok()
            }
            // SAFETY: the view owns the word, crossed at `held`, and is
            // borrowed for the call; a shared parameter writes nothing.
            held => unsafe { lend(self.rt, interner, word, held, f) }.ok(),
        }
    }

    /// As `with`, exclusively: the write stays in the view's own word, which
    /// is released when the call ends, except through a `&mut` argument,
    /// whose target is the caller's. `None` at a `&` argument.
    pub fn with_mut<Q, O, F>(&mut self, i: usize, f: F) -> Option<O>
    where
        F: Borrows<Rt, Q, O>,
        F::Marker: Lendable<Rt, Loan = Mut>,
    {
        let holder = self.words.as_mut().get_mut(i)?;
        let interner = &self.site.interner;
        match &self.site.tys[i] {
            Ty::Ref(Mutability::Shared, _) => None,
            Ty::Ref(Mutability::Mut, target) => {
                let TypeArg::Uniform(target) = &**target else {
                    return None;
                };
                // SAFETY: the word was crossed at `&mut target`, so it is the
                // one live name of a storage of `target` the caller lent the
                // call exclusively.
                unsafe { lend_through(self.rt, interner, holder, target, f) }.ok()
            }
            held => {
                // SAFETY: `&mut self` names the view exclusively, and a lent
                // parameter writes inside the storage the word names or the
                // word in place, never another holder's.
                let word = unsafe { holder.value_mut(Holding::new()) };
                // SAFETY: the word was crossed at `held`, and `word` is its
                // only live name for the call.
                unsafe { lend(self.rt, interner, word, held, f) }.ok()
            }
        }
    }

    /// Every argument laid out by its settled type, in order (RFC-0033). A
    /// reference is refused: its target is the caller's storage, which
    /// bytes do not carry.
    pub fn encode(&self) -> Result<Encoded, SpaceError> {
        let mut out = Vec::new();
        for (at, (ty, word)) in self.site.tys.iter().zip(self.words.as_ref()).enumerate() {
            let laid = match ty {
                Ty::Ref(..) => Err(SpaceError::new("a reference names the caller's storage")),
                // SAFETY: the view owns the word, crossed at `ty`.
                ty => unsafe { self.rt.encode(ty, word, &mut out) },
            };
            laid.map_err(|refused| {
                SpaceError::new(format!(
                    "argument {at} of type {} is not encoded: {refused}",
                    ty.display(&self.site.interner)
                ))
            })?;
        }
        Ok(Encoded(out.into_boxed_slice()))
    }
}

// SAFETY: the view takes each of its words by `Args::moved_in`, over the
// run of by-value words its marker occupies; it crosses nothing else, and
// keeps the capability's runtime for lending alone.
unsafe impl<'a, 'w, P, Rt> Takes<'a, 'w, ByArgs<P>, Rt> for Args<'a, P, Rt>
where
    P: Members + 'static,
    Rt: Runtime,
{
    unsafe fn take(rt: Crossing<'a, Rt>, run: &'a [Rt::Value], site: &'a ArgsSite) -> Self {
        // SAFETY: the caller's contract: `run` is this parameter's own words
        // of the call, each moved to the callee at the type the checker
        // settled, which is what the site table holds.
        unsafe { Args::moved_in(rt, run, site) }
    }
}
