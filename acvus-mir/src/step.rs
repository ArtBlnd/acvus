//! The per-element step a stream adaptor or consumer states (RFC-0099
//! rule 1). `#[extern_fn(step(..))]` writes it, `optimize::fusion` reads
//! it through the instance a call names, and `acvus-ext`'s differential
//! harness samples the promise that the handler and the step agree.
//!
//! A parameter is numbered as the declaration's acvus parameters are, which
//! is the order of a call's arguments.

use acvus_ast::Literal;
use acvus_utils::QualifiedRef;

use crate::ty::Mutability;

#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    Adaptor {
        stream: StreamParam,
        flow: AdaptorFlow,
    },
    Consumer {
        stream: StreamParam,
        state: Option<Term>,
        body: ConsumerBlock,
        finish: Term,
    },
}

/// The `Instance` parameter a step pulls, and the index of the requirement
/// that owns it, which a call's `Callee::Extern::required` stands one to
/// one against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamParam {
    pub param: usize,
    pub requirement: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AdaptorFlow {
    Yield(Term),
    Skip,
    Done,
    /// Each element of the container, in order, then the next element.
    Nest(Term),
    If {
        cond: Term,
        then: Box<AdaptorFlow>,
        otherwise: Box<AdaptorFlow>,
    },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConsumerBlock {
    pub stmts: Vec<ConsumerStmt>,
    pub breaks: Option<Term>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConsumerStmt {
    Set(Term),
    Run {
        name: QualifiedRef,
        args: Vec<Term>,
    },
    If {
        cond: Term,
        then: ConsumerBlock,
        otherwise: ConsumerBlock,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Elem,
    State,
    ValueParam(usize),
    CallClosure {
        param: usize,
        args: Vec<Term>,
    },
    /// Named as RFC-0082 rule 2 names an extern, at the instance its
    /// arguments' types choose.
    CallExtern {
        name: QualifiedRef,
        args: Vec<Term>,
    },
    Const(Literal),
    Lend(Mutability, Box<Term>),
    /// `Add(Overflow::Wrap)`: what `acvus-ext`'s `Num::add` computes for
    /// `sum`, wrapping at an integer width and IEEE addition at `f64`.
    WrappingAdd { left: Box<Term>, right: Box<Term> },
}

impl Step {
    pub fn stream(&self) -> StreamParam {
        match self {
            Step::Adaptor { stream, .. } | Step::Consumer { stream, .. } => *stream,
        }
    }

    pub fn terms(&self) -> Vec<&Term> {
        match self {
            Step::Adaptor { flow, .. } => flow.terms(),
            Step::Consumer {
                state,
                body,
                finish,
                ..
            } => state.iter().chain(body.terms()).chain([finish]).collect(),
        }
    }

    pub fn named_externs(&self) -> Vec<QualifiedRef> {
        let runs = match self {
            Step::Adaptor { .. } => Vec::new(),
            Step::Consumer { body, .. } => body.runs(),
        };
        let mut named: Vec<QualifiedRef> = runs.iter().map(|run| run.name).collect();
        for term in self.terms() {
            term.visit(&mut |term| {
                if let Term::CallExtern { name, .. } = term {
                    named.push(*name);
                }
            });
        }
        named
    }

    pub fn called_closures(&self) -> Vec<usize> {
        closures_called(self.terms())
    }
}

pub fn closures_called<'t>(terms: impl IntoIterator<Item = &'t Term>) -> Vec<usize> {
    let mut called = Vec::new();
    for term in terms {
        term.visit(&mut |term| {
            if let Term::CallClosure { param, .. } = term {
                called.push(*param);
            }
        });
    }
    called.sort_unstable();
    called.dedup();
    called
}

/// A `ConsumerStmt::Run`, wherever it stands in the block.
#[derive(Debug, Clone, Copy)]
pub struct RunCall<'b> {
    pub name: QualifiedRef,
    pub args: &'b [Term],
}

impl AdaptorFlow {
    pub fn terms(&self) -> Vec<&Term> {
        let mut found = Vec::new();
        self.collect_terms(&mut found);
        found
    }

    fn collect_terms<'a>(&'a self, found: &mut Vec<&'a Term>) {
        match self {
            AdaptorFlow::Yield(term) | AdaptorFlow::Nest(term) => found.push(term),
            AdaptorFlow::Skip | AdaptorFlow::Done => {}
            AdaptorFlow::If {
                cond,
                then,
                otherwise,
            } => {
                found.push(cond);
                then.collect_terms(found);
                otherwise.collect_terms(found);
            }
        }
    }
}

impl ConsumerBlock {
    pub fn terms(&self) -> Vec<&Term> {
        let mut found = Vec::new();
        self.collect_terms(&mut found);
        found
    }

    pub fn runs(&self) -> Vec<RunCall<'_>> {
        let mut found = Vec::new();
        self.collect_runs(&mut found);
        found
    }

    fn collect_runs<'a>(&'a self, found: &mut Vec<RunCall<'a>>) {
        for stmt in &self.stmts {
            match stmt {
                ConsumerStmt::Set(_) => {}
                ConsumerStmt::Run { name, args } => found.push(RunCall { name: *name, args }),
                ConsumerStmt::If {
                    then, otherwise, ..
                } => {
                    then.collect_runs(found);
                    otherwise.collect_runs(found);
                }
            }
        }
    }

    fn collect_terms<'a>(&'a self, found: &mut Vec<&'a Term>) {
        for stmt in &self.stmts {
            match stmt {
                ConsumerStmt::Set(term) => found.push(term),
                ConsumerStmt::Run { args, .. } => found.extend(args),
                ConsumerStmt::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    found.push(cond);
                    then.collect_terms(found);
                    otherwise.collect_terms(found);
                }
            }
        }
        found.extend(&self.breaks);
    }
}

impl Term {
    pub fn visit<'t>(&'t self, on: &mut impl FnMut(&'t Term)) {
        on(self);
        match self {
            Term::Elem | Term::State | Term::ValueParam(_) | Term::Const(_) => {}
            Term::CallClosure { args, .. } | Term::CallExtern { args, .. } => {
                for arg in args {
                    arg.visit(on);
                }
            }
            Term::Lend(_, term) => term.visit(on),
            Term::WrappingAdd { left, right } => {
                left.visit(on);
                right.visit(on);
            }
        }
    }
}
