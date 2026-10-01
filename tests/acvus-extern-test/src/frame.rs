//! A frame of the run `WordRuntime` stands in for: the values a program's
//! variables hold, and the calls it makes with them through an entry, as the
//! runtime's boundary does (runtime contract §2, The entry).
//!
//! The frame is where the checker's decisions come in, so it trusts its
//! caller as a run trusts the checker: each `Pass` is how the call site passes
//! that argument, and it must be what the handler's parameter mode opens.

use acvus_extern::done::Done;
use acvus_extern::register::{AwaitedFn, ExternFn};
use acvus_extern::{Entry, Primitive, SiteShape};

use crate::boundary::{Held, Word};
use crate::runtime::WordRuntime;

/// How one argument is passed.
#[derive(Debug, Clone, Copy)]
pub enum Pass {
    /// The variable's value, handed over: the call owns it from here on.
    Give(usize),
    /// The variable lent in place.
    Lend(usize),
    /// The variable's value, a reference, passed to a call that reads what it
    /// names; the variable still owns it.
    Same(usize),
    /// The variable's value, a function, passed to a call that calls it; the
    /// variable still owns it.
    Function(usize),
}

/// A frame's variables, each boxed so that a lent one keeps its address.
pub struct Frame<'r> {
    runtime: &'r WordRuntime,
    variables: Vec<Option<Box<Held>>>,
}

impl<'r> Frame<'r> {
    pub fn new(runtime: &'r WordRuntime) -> Self {
        Frame { runtime, variables: Vec::new() }
    }

    /// Calls `glue` with the arguments `passes` names, and keeps its result
    /// in a new variable, whose index it returns.
    pub fn call<const N: usize>(
        &mut self,
        passes: [Pass; N],
        glue: impl FnOnce(Entry<'_, WordRuntime, N>) -> Done<WordRuntime>,
    ) -> usize {
        let args = self.arguments(passes);
        let site = Self::site();

        let done = glue(Entry::new(self.runtime, args, &site));

        self.keep(done)
    }

    /// As `call`, for a glue whose body is awaited.
    pub async fn call_awaited<const N: usize>(
        &mut self,
        passes: [Pass; N],
        glue: impl AsyncFnOnce(Entry<'_, WordRuntime, N>) -> Done<WordRuntime>,
    ) -> usize {
        let args = self.arguments(passes);
        let site = Self::site();

        let done = glue(Entry::new(self.runtime, args, &site)).await;

        self.keep(done)
    }

    /// Calls the Rust function the variable `function` holds with the
    /// arguments `passes` names, as a script calls a function value, and
    /// keeps its result in a new variable, whose index it returns.
    pub fn call_function<const N: usize>(&mut self, function: usize, passes: [Pass; N]) -> usize {
        let args = self.arguments(passes);
        let site = Self::site();

        let callee = self.variables[function].as_deref().expect("a function given away is not called");
        let done = callee.function()(self.runtime, &args, &site);

        self.keep(done)
    }

    /// A variable holding a script function value, the function `f`: the
    /// run's own way to make one, as a program makes a lambda. A function
    /// value is called with a run of the arity its type says, which this
    /// runtime, holding every function value at one Rust type, reads back.
    pub fn function<F, const N: usize>(&mut self, f: F) -> usize
    where
        F: ExternFn<WordRuntime, N>,
    {
        let value = crate::boundary::closure(Box::new(move |runtime, run: &[Word], site| {
            f.call(runtime, run.try_into().expect("a function value is called at its own arity"), site)
        }));
        self.variables.push(Some(Box::new(Held::kept(value))));
        self.variables.len() - 1
    }

    /// A variable holding a program's array of `elements`: the run's own way
    /// to make one, as a program writes an array literal.
    pub fn array<P>(&mut self, elements: &[P]) -> usize
    where
        P: Primitive,
    {
        let value = crate::boundary::array(elements.iter().map(|&e| crate::boundary::scalar(e)).collect());
        self.variables.push(Some(Box::new(Held::kept(value))));
        self.variables.len() - 1
    }

    /// Calls the awaited function `f` through its dispatch with the
    /// arguments `passes` names, and keeps its result in a new variable.
    pub async fn call_dispatch_awaited<F, const N: usize>(&mut self, f: &F, passes: [Pass; N]) -> usize
    where
        F: AwaitedFn<WordRuntime, N>,
    {
        let args = self.arguments(passes);
        let site = Self::site();

        let done = f.call(self.runtime, args, &site).await;

        self.keep(done)
    }

    /// What the variable, an `Option<&P>`, names: the target of its part,
    /// copied out, where it is present.
    pub fn option_referent<P>(&self, variable: usize) -> Option<P::Owned>
    where
        P: ?Sized + ToOwned,
    {
        self.variables[variable].as_deref().expect("a variable given away is not read").option_referent::<P>()
    }

    /// What the variable, a pair of references to `P`, names, copied out.
    pub fn pair_referents<P>(&self, variable: usize) -> (P::Owned, P::Owned)
    where
        P: ?Sized + ToOwned,
    {
        self.variables[variable].as_deref().expect("a variable given away is not read").pair_referents::<P>()
    }

    fn arguments<const N: usize>(&mut self, passes: [Pass; N]) -> [Word; N] {
        passes.map(|pass| match pass {
            Pass::Give(i) => self.variables[i].take().expect("a variable is given once").hand_over(),
            Pass::Lend(i) => self.variable(i).lend(),
            Pass::Same(i) | Pass::Function(i) => self.variable(i).same(),
        })
    }

    fn site() -> SiteShape<WordRuntime> {
        SiteShape::new(Box::new([]))
    }

    fn keep(&mut self, done: Done<WordRuntime>) -> usize {
        self.variables.push(Some(Box::new(Held::of_the_result(done))));
        self.variables.len() - 1
    }

    fn variable(&mut self, i: usize) -> &mut Held {
        self.variables[i].as_mut().expect("a variable given away is not passed again")
    }
}
