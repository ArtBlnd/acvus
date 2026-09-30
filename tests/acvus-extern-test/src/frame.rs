//! A frame of the run `WordRuntime` stands in for: the values a program's
//! variables hold, and the calls it makes with them through an entry, as the
//! runtime's boundary does (runtime contract §2, The entry).
//!
//! The frame is where the checker's decisions come in, so it trusts its
//! caller as a run trusts the checker: each `Pass` is how the call site passes
//! that argument, and it must be what the handler's parameter mode opens. A
//! debug build's entry asserts that they agree.

use acvus_extern::done::Done;
use acvus_extern::register::{ExternFn, Registers};
use acvus_extern::runtime::{Mutability, Runtime};
use acvus_utils::QualifiedRef;
use acvus_extern::{Entry, Primitive, SiteShape};

use crate::boundary::{Held, Word};

/// How one argument is passed.
#[derive(Debug, Clone, Copy)]
pub enum Pass {
    /// The variable's value, handed over: the call owns it from here on.
    Give(usize),
    /// The variable lent in place.
    Lend(usize, Mutability),
    /// The variable's value, a reference, passed to a call that reads what it
    /// names; the variable still owns it.
    Same(usize, Mutability),
    /// The variable's value, a function, passed to a call that calls it; the
    /// variable still owns it.
    Function(usize),
}

impl Pass {
    fn lent(self) -> Option<Mutability> {
        match self {
            Pass::Give(_) | Pass::Function(_) => None,
            Pass::Lend(_, m) | Pass::Same(_, m) => Some(m),
        }
    }
}

/// A frame's variables, each boxed so that a lent one keeps its address.
pub struct Frame<'r, R> {
    runtime: &'r R,
    variables: Vec<Option<Box<Held>>>,
}

impl<'r, R> Frame<'r, R>
where
    R: Runtime<Value = Word>,
{
    pub fn new(runtime: &'r R) -> Self {
        Frame { runtime, variables: Vec::new() }
    }

    /// Calls `glue` with the arguments `passes` names, and keeps its result
    /// in a new variable, whose index it returns.
    pub fn call(&mut self, passes: &[Pass], glue: impl FnOnce(Entry<'_, R>) -> Done<R>) -> usize {
        let args = self.arguments(passes);
        let site = Self::site(passes);

        let done = glue(Entry::new(self.runtime, &args, &site));

        self.keep(done)
    }

    /// As `call`, for a glue whose body is awaited.
    pub async fn call_awaited(&mut self, passes: &[Pass], glue: impl AsyncFnOnce(Entry<'_, R>) -> Done<R>) -> usize {
        let args = self.arguments(passes);
        let site = Self::site(passes);

        let done = glue(Entry::new(self.runtime, &args, &site)).await;

        self.keep(done)
    }

    /// Calls the Rust function the variable `function` holds with the
    /// arguments `passes` names, as a script calls a function value, and
    /// keeps its result in a new variable, whose index it returns.
    pub fn call_function(&mut self, function: usize, passes: &[Pass]) -> usize {
        let args = self.arguments(passes);
        let site = Self::site(passes);

        let callee = self.variables[function].as_deref().expect("a function given away is not called");
        let done = callee.function::<R>().call(self.runtime, &args, &site);

        self.keep(done)
    }

    /// A variable holding a script function value, the function `f`: the
    /// run's own way to make one, as a program makes a lambda.
    pub fn function(&mut self, f: impl ExternFn<R>) -> usize {
        let value = crate::boundary::closure::<R>(Box::new(f));
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

    /// Calls the function registered at `path` with the arguments `passes`
    /// names, as a compiled call does once its path is resolved, and keeps
    /// its result in a new variable, whose index it returns.
    pub fn call_registered(&mut self, path: QualifiedRef, passes: &[Pass]) -> usize
    where
        R: Registers,
    {
        let args = self.arguments(passes);
        let site = Self::site(passes);

        let f = self.runtime.registered(path).expect("a compiled call names a registered function");
        let done = f.call(self.runtime, &args, &site);

        self.keep(done)
    }

    /// Calls the awaited function registered at `path` with the arguments
    /// `passes` names, and keeps its result in a new variable.
    pub async fn call_registered_awaited(&mut self, path: QualifiedRef, passes: &[Pass]) -> usize
    where
        R: Registers,
    {
        let args = self.arguments(passes);
        let site = Self::site(passes);

        let call = self.runtime.call_awaited(path, &args, &site).expect("a compiled call names a registered function");
        let done = call.await;

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

    fn arguments(&mut self, passes: &[Pass]) -> Vec<Word> {
        let mut args = Vec::with_capacity(passes.len());
        for &pass in passes {
            let arg = match pass {
                Pass::Give(i) => self.variables[i].take().expect("a variable is given once").hand_over(),
                Pass::Lend(i, _) => self.variable(i).lend(),
                Pass::Same(i, _) | Pass::Function(i) => self.variable(i).same(),
            };
            args.push(arg);
        }
        args
    }

    fn site(passes: &[Pass]) -> SiteShape<R> {
        SiteShape::new(passes.iter().map(|p| p.lent()).collect(), Box::new([]))
    }

    fn keep(&mut self, done: Done<R>) -> usize {
        self.variables.push(Some(Box::new(Held::of_the_result(done))));
        self.variables.len() - 1
    }

    fn variable(&mut self, i: usize) -> &mut Held {
        self.variables[i].as_mut().expect("a variable given away is not passed again")
    }
}
